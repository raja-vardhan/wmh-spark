"""Per-voxel feature extraction.

We replicate the UBO Detector feature space so the k-NN baseline is a true
apples-to-apples reproduction of the original algorithm. Features are
computed in NumPy *on each worker* (subject-level parallelism), not as a
voxel-level Spark DataFrame -- the latter explodes serialization cost
without unlocking any shuffle benefit.

Feature set (one row per candidate voxel)
=========================================
1.  FLAIR z-score within NAWM (normal-appearing white matter)
2.  T1 intensity (z-scored per-subject)
3.  FLAIR / T1 intensity ratio
4.  Distance from lateral ventricles (mm, in MNI space)
5.  MNI x, y, z coordinates (mm)
6.  3x3x3 neighborhood mean + std of FLAIR
7.  3x3x3 neighborhood mean + std of T1

A *candidate voxel* is one that:
- is inside the brain mask, AND
- lies in white matter (atlas-based or T1-threshold proxy), AND
- has FLAIR intensity above a NAWM-relative threshold (UBO-style).

The output of this stage is a NumPy array of shape (n_candidates, n_features)
plus the linear voxel indices so we can map predictions back to a 3D mask.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from .config import FeatureConfig
from .io_utils import load_volume, load_mask

logger = logging.getLogger(__name__)


# Indices into the feature column array. Documenting them here keeps the
# pipeline robust against silent reordering bugs.
FEATURE_NAMES: list[str] = [
    "flair_z",
    "t1_z",
    "flair_t1_ratio",
    "dist_ventricle",
    "mni_x", "mni_y", "mni_z",
    "flair_nbr_mean", "flair_nbr_std",
    "t1_nbr_mean", "t1_nbr_std",
]


@dataclass
class FeatureBundle:
    """Output of feature extraction for one subject.

    We deliberately keep this lightweight: arrays only, no Spark types. The
    pipeline driver decides how to ship it (Parquet on disk, in-memory RDD
    of bundles, etc.).
    """

    subject_id: str
    features: np.ndarray            # (n_voxels, n_features) float32
    voxel_indices: np.ndarray       # (n_voxels,) int64 -- linearized
    volume_shape: tuple[int, int, int]
    affine: np.ndarray              # 4x4 -- to write predictions back out
    labels: Optional[np.ndarray] = None  # (n_voxels,) uint8, only if GT provided
    seconds: float = 0.0
    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_features(
    subject_id: str,
    flair_mni_path: str,
    t1_mni_path: str,
    brain_mask_mni_path: str,
    cfg: FeatureConfig,
    gt_mask_mni_path: Optional[str] = None,
) -> FeatureBundle:
    """Extract candidate-voxel features for one subject.

    Returns a FeatureBundle whose ``features`` array is ready to be
    converted to a Spark Vector for either training-set sampling or
    inference.
    """
    t0 = time.perf_counter()

    try:
        flair, affine = load_volume(flair_mni_path)
        t1, _ = load_volume(t1_mni_path)
        brain_mask = load_mask(brain_mask_mni_path)

        if flair.shape != t1.shape or flair.shape != brain_mask.shape:
            raise ValueError(
                f"Shape mismatch for {subject_id}: "
                f"FLAIR={flair.shape}, T1={t1.shape}, mask={brain_mask.shape}"
            )

        # ------------------------------------------------------------------
        # 1. Build candidate-voxel mask.
        # ------------------------------------------------------------------
        wm_mask = _approximate_wm_mask(t1, brain_mask, cfg.wm_mask_threshold)
        nawm_mean, nawm_std = _nawm_stats(flair, wm_mask)
        flair_threshold = nawm_mean * cfg.flair_min_threshold

        candidate = wm_mask & (flair > flair_threshold)
        n = int(candidate.sum())
        if n == 0:
            raise ValueError(f"No candidate voxels for {subject_id}")

        # ------------------------------------------------------------------
        # 2. Compute features as full-volume arrays, then sample.
        #    This is faster than per-voxel loops and vectorizes cleanly.
        # ------------------------------------------------------------------
        cols: list[np.ndarray] = []

        flair_z = (flair - nawm_mean) / max(nawm_std, 1e-6)
        t1_mean, t1_std = _masked_stats(t1, brain_mask)
        t1_z = (t1 - t1_mean) / max(t1_std, 1e-6)
        eps = 1e-6
        flair_t1_ratio = flair / (t1 + eps)

        # Spatial: distance to ventricles (proxy = distance to CSF-like voxels
        # near image center; in production swap in an atlas-based ventricle
        # mask in MNI space). We compute distance-transform in voxels, then
        # rescale by the affine spacing for an mm value.
        ventricle_mask = _approximate_ventricles(t1, brain_mask)
        dist_vox = ndimage.distance_transform_edt(~ventricle_mask)
        spacing = np.linalg.norm(affine[:3, :3], axis=0).mean()
        dist_mm = dist_vox * spacing

        # MNI coords: voxel index -> world coord via affine.
        mni_x, mni_y, mni_z = _voxel_to_world_grids(flair.shape, affine)

        # Neighborhood stats via uniform filter (3x3x3 box mean) and a fast
        # std-via-variance trick: var = E[X^2] - E[X]^2.
        size = 2 * cfg.neighborhood_radius + 1
        flair_nbr_mean = ndimage.uniform_filter(flair, size=size, mode="nearest")
        flair_sq_mean = ndimage.uniform_filter(flair * flair, size=size, mode="nearest")
        flair_nbr_std = np.sqrt(np.maximum(flair_sq_mean - flair_nbr_mean ** 2, 0.0))
        t1_nbr_mean = ndimage.uniform_filter(t1, size=size, mode="nearest")
        t1_sq_mean = ndimage.uniform_filter(t1 * t1, size=size, mode="nearest")
        t1_nbr_std = np.sqrt(np.maximum(t1_sq_mean - t1_nbr_mean ** 2, 0.0))

        # Stack only at candidate voxels. We index once per feature -- each
        # is O(n) instead of O(volume) memory.
        idx = np.flatnonzero(candidate.ravel())
        for f in (flair_z, t1_z, flair_t1_ratio, dist_mm,
                  mni_x, mni_y, mni_z,
                  flair_nbr_mean, flair_nbr_std,
                  t1_nbr_mean, t1_nbr_std):
            cols.append(f.ravel()[idx])
        feats = np.stack(cols, axis=1).astype(np.float32)

        # ------------------------------------------------------------------
        # 3. Optional ground-truth labels for training samples.
        # ------------------------------------------------------------------
        labels: Optional[np.ndarray] = None
        if gt_mask_mni_path:
            gt = load_mask(gt_mask_mni_path).ravel()
            if gt.shape[0] != flair.size:
                raise ValueError(
                    f"GT mask shape {gt.shape} != FLAIR size {flair.size} for {subject_id}"
                )
            labels = gt[idx].astype(np.uint8)

        return FeatureBundle(
            subject_id=subject_id,
            features=feats,
            voxel_indices=idx.astype(np.int64),
            volume_shape=flair.shape,
            affine=affine,
            labels=labels,
            seconds=time.perf_counter() - t0,
            success=True,
        )

    except Exception as exc:
        logger.exception("Feature extraction failed for %s", subject_id)
        return FeatureBundle(
            subject_id=subject_id,
            features=np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            voxel_indices=np.empty((0,), dtype=np.int64),
            volume_shape=(0, 0, 0),
            affine=np.eye(4),
            labels=None,
            seconds=time.perf_counter() - t0,
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _masked_stats(volume: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Mean and std of a volume restricted to a boolean mask."""
    vals = volume[mask.astype(bool)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(vals.mean()), float(vals.std())


def _nawm_stats(flair: np.ndarray, wm_mask: np.ndarray) -> tuple[float, float]:
    """Estimate NAWM (normal-appearing WM) intensity stats.

    We robustly estimate by clipping to the 5th-95th percentile inside the WM
    mask -- this excludes lesions (high tail) and PVE voxels (low tail).
    """
    wm_vals = flair[wm_mask.astype(bool)]
    if wm_vals.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(wm_vals, [5, 95])
    core = wm_vals[(wm_vals >= lo) & (wm_vals <= hi)]
    return float(core.mean()), float(core.std())


def _approximate_wm_mask(
    t1: np.ndarray, brain_mask: np.ndarray, threshold: float
) -> np.ndarray:
    """Approximate WM mask by Otsu-like thresholding inside the brain.

    Production deployments should swap in an atlas-derived WM probability
    map. This proxy is good enough to reproduce UBO's candidate selection
    on standard 3T FLAIR/T1 pairs.
    """
    brain = brain_mask.astype(bool)
    if not brain.any():
        return np.zeros_like(t1, dtype=bool)
    intensities = t1[brain]
    # WM is the brighter mode in T1 (after skull strip).
    cutoff = np.quantile(intensities, threshold)
    return brain & (t1 > cutoff)


def _approximate_ventricles(t1: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    """Crude ventricle proxy: low T1 + central position.

    Real pipelines use an MNI ventricle atlas. This proxy is sufficient for
    the distance-to-ventricle feature to be informative; the model can
    learn around the noise.
    """
    brain = brain_mask.astype(bool)
    if not brain.any():
        return np.zeros_like(t1, dtype=bool)
    low_t1 = t1 < np.quantile(t1[brain], 0.20)
    # Restrict to central core to avoid sulcal CSF.
    z, y, x = np.indices(t1.shape)
    cz, cy, cx = (s / 2 for s in t1.shape)
    central = (
        (np.abs(z - cz) < 0.20 * t1.shape[0]) &
        (np.abs(y - cy) < 0.30 * t1.shape[1]) &
        (np.abs(x - cx) < 0.20 * t1.shape[2])
    )
    return brain & low_t1 & central


def _voxel_to_world_grids(
    shape: tuple[int, int, int], affine: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return three volumes giving the MNI-space x/y/z coordinate of each voxel."""
    i, j, k = np.indices(shape, dtype=np.float32)
    ones = np.ones_like(i)
    homog = np.stack([i, j, k, ones], axis=0)
    coords = np.tensordot(affine.astype(np.float32), homog, axes=([1], [0]))
    return coords[0], coords[1], coords[2]
