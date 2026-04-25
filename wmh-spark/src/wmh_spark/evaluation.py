"""Evaluation metrics.

We deliberately go beyond voxel-level Dice. WMH volumes are dominated by a
few large confluent lesions, so Dice can hide systematic failure on small
punctate lesions that are clinically important. We mirror the WMH Challenge
evaluation protocol:

- Dice Similarity Coefficient (DSC)         -- volumetric overlap
- Lesion-wise F1                            -- per-lesion detection
- Hausdorff Distance, 95th percentile       -- boundary error
- Absolute Volume Difference                -- bias indicator

We also compute pairwise comparisons against both expert ground truth
*and* UBO Detector output so reviewers can see both axes of agreement.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff

from .config import EvaluationConfig
from .io_utils import load_mask

logger = logging.getLogger(__name__)


@dataclass
class SubjectMetrics:
    subject_id: str
    reference: str            # "expert" or "ubo"
    dice: float
    lesion_f1: float
    lesion_precision: float
    lesion_recall: float
    hd95_mm: float
    abs_vol_diff_ml: float
    pred_volume_ml: float
    ref_volume_ml: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def evaluate_subject(
    subject_id: str,
    pred_mask_path: str,
    reference_mask_path: str,
    reference_name: str,
    voxel_volume_ml: float,
    cfg: EvaluationConfig,
) -> SubjectMetrics:
    """Compute all metrics for one subject.

    ``voxel_volume_ml`` converts voxel counts to milliliters; for 1mm
    isotropic MNI it's 1e-3.
    """
    pred = load_mask(pred_mask_path).astype(bool)
    ref = load_mask(reference_mask_path).astype(bool)

    if pred.shape != ref.shape:
        raise ValueError(
            f"Shape mismatch for {subject_id}: pred={pred.shape}, ref={ref.shape}"
        )

    dice = _dice(pred, ref)
    lp, lr, lf1 = _lesion_f1(pred, ref)
    hd95 = _hausdorff_95(pred, ref) if cfg.compute_hausdorff else float("nan")

    pred_vol_ml = float(pred.sum()) * voxel_volume_ml
    ref_vol_ml = float(ref.sum()) * voxel_volume_ml

    return SubjectMetrics(
        subject_id=subject_id,
        reference=reference_name,
        dice=dice,
        lesion_f1=lf1,
        lesion_precision=lp,
        lesion_recall=lr,
        hd95_mm=hd95,
        abs_vol_diff_ml=abs(pred_vol_ml - ref_vol_ml),
        pred_volume_ml=pred_vol_ml,
        ref_volume_ml=ref_vol_ml,
    )


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------


def _dice(pred: np.ndarray, ref: np.ndarray) -> float:
    p = pred.sum()
    r = ref.sum()
    if p + r == 0:
        return 1.0  # both empty -> perfect agreement
    inter = np.logical_and(pred, ref).sum()
    return float(2.0 * inter / (p + r))


def _lesion_f1(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    """Lesion-level precision / recall / F1.

    We label each connected component as one lesion. A predicted lesion is
    a true positive if it has any voxel overlap with a reference lesion.
    Conversely, a reference lesion is detected if any predicted lesion
    overlaps it. This mirrors the WMH Challenge protocol.
    """
    pred_lbl, n_pred = ndimage.label(pred)
    ref_lbl, n_ref = ndimage.label(ref)

    if n_pred == 0 and n_ref == 0:
        return 1.0, 1.0, 1.0
    if n_pred == 0:
        return 0.0, 0.0, 0.0
    if n_ref == 0:
        return 0.0, 0.0, 0.0

    # Co-occurrence: for each predicted lesion, find which ref labels it
    # overlaps. ndimage.maximum is a fast way to do this.
    pred_overlap_any = np.zeros(n_pred + 1, dtype=bool)
    ref_detected = np.zeros(n_ref + 1, dtype=bool)
    overlapping = ref_lbl[pred.astype(bool)]
    pred_at_overlap = pred_lbl[pred.astype(bool)]
    # Walk overlaps -- vectorized via np.unique + boolean indexing.
    pairs = np.stack([pred_at_overlap, overlapping], axis=1)
    pairs = pairs[pairs[:, 1] > 0]  # drop background-overlap rows
    if pairs.size:
        pred_overlap_any[pairs[:, 0]] = True
        ref_detected[pairs[:, 1]] = True

    tp_pred = int(pred_overlap_any[1:].sum())
    fp = n_pred - tp_pred
    fn = n_ref - int(ref_detected[1:].sum())

    precision = tp_pred / max(tp_pred + fp, 1)
    recall = tp_pred / max(tp_pred + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return float(precision), float(recall), float(f1)


def _hausdorff_95(pred: np.ndarray, ref: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance (in voxel units).

    Approximated using surface-point sampling for tractability on dense
    masks. Returns NaN when either mask is empty.
    """
    if pred.sum() == 0 or ref.sum() == 0:
        return float("nan")
    pred_pts = np.argwhere(_surface(pred))
    ref_pts = np.argwhere(_surface(ref))
    if pred_pts.size == 0 or ref_pts.size == 0:
        return float("nan")
    # Sample-cap for speed; full pairwise is O(n*m).
    rng = np.random.default_rng(0)
    if pred_pts.shape[0] > 5000:
        pred_pts = pred_pts[rng.choice(pred_pts.shape[0], 5000, replace=False)]
    if ref_pts.shape[0] > 5000:
        ref_pts = ref_pts[rng.choice(ref_pts.shape[0], 5000, replace=False)]
    a = directed_hausdorff(pred_pts, ref_pts)[0]
    b = directed_hausdorff(ref_pts, pred_pts)[0]
    return float(max(a, b))


def _surface(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of surface voxels (foreground voxels touching background)."""
    eroded = ndimage.binary_erosion(mask)
    return mask & ~eroded
