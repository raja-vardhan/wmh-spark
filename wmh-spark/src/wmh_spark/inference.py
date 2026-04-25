"""Inference stage: broadcast model, predict per subject on workers.

Each worker receives the trained model via Spark broadcast, then for every
subject in its partition:
1. Loads the FeatureBundle (or recomputes features).
2. Runs predict_proba on the candidate-voxel feature matrix.
3. Reconstructs the 3D probability volume and writes a binary mask using
   the configured threshold.
4. Returns a small per-subject result record (paths + timing).

This keeps volumes off the driver and exploits subject-level parallelism.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .features import FeatureBundle
from .io_utils import save_volume

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    subject_id: str
    prob_path: str
    mask_path: str
    n_voxels: int
    n_predicted_lesion: int
    seconds: float
    success: bool
    error: Optional[str] = None


def infer_subject(
    bundle: FeatureBundle,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    threshold: float,
    out_dir: str,
) -> InferenceResult:
    """Run the broadcast model on one subject's features and write outputs.

    ``predict_fn`` is a closure capturing the broadcast model bundle. We
    pass it in rather than the model itself so this function is agnostic
    to which classifier (k-NN / RF / XGB) is in use.
    """
    t0 = time.perf_counter()

    if not bundle.success:
        return InferenceResult(
            subject_id=bundle.subject_id,
            prob_path="", mask_path="",
            n_voxels=0, n_predicted_lesion=0,
            seconds=time.perf_counter() - t0,
            success=False, error="bad feature bundle",
        )

    out_dir_p = Path(out_dir) / bundle.subject_id
    out_dir_p.mkdir(parents=True, exist_ok=True)

    try:
        proba = predict_fn(bundle.features)

        # Reconstruct full-volume probability map by scattering predictions
        # back into the original voxel grid.
        prob_vol = np.zeros(int(np.prod(bundle.volume_shape)), dtype=np.float32)
        prob_vol[bundle.voxel_indices] = proba
        prob_vol = prob_vol.reshape(bundle.volume_shape)

        mask_vol = (prob_vol >= threshold).astype(np.uint8)

        prob_path = out_dir_p / "wmh_prob.nii.gz"
        mask_path = out_dir_p / "wmh_mask.nii.gz"
        save_volume(prob_vol, bundle.affine, prob_path)
        save_volume(mask_vol, bundle.affine, mask_path, dtype=np.uint8)

        return InferenceResult(
            subject_id=bundle.subject_id,
            prob_path=str(prob_path),
            mask_path=str(mask_path),
            n_voxels=int(bundle.features.shape[0]),
            n_predicted_lesion=int(mask_vol.sum()),
            seconds=time.perf_counter() - t0,
            success=True,
        )
    except Exception as exc:
        logger.exception("Inference failed for %s", bundle.subject_id)
        return InferenceResult(
            subject_id=bundle.subject_id,
            prob_path="", mask_path="",
            n_voxels=0, n_predicted_lesion=0,
            seconds=time.perf_counter() - t0,
            success=False, error=str(exc),
        )
