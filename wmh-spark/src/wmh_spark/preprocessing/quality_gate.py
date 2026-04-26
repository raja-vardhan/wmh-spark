"""Quality gate for HD-BET skull-stripped output validation.

Phase 1 (smoke-test): structural assertions — binary mask, shape/affine match,
    foreground fraction in [5%, 95%].
Phase 2 (Kaggle): DSC > 0.85 against expert reference brain masks.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from wmh_spark.io_utils import load_mask, load_volume

logger = logging.getLogger(__name__)

DSC_THRESHOLD = 0.85
FOREGROUND_MIN = 0.05
FOREGROUND_MAX = 0.95


class QualityGateError(Exception):
    pass


class QualityGate:
    def __init__(self, smoke_test: bool = False) -> None:
        self.smoke_test = smoke_test

    def assert_structural(self, mask_path: Path, raw_input_path: Path) -> None:
        """Smoke-test quality check: validate structural properties of the brain mask.

        Checks:
          1. All mask values are binary (0 or 1).
          2. Mask shape matches raw input shape.
          3. Mask affine matches raw input affine.
          4. Foreground fraction is in [FOREGROUND_MIN, FOREGROUND_MAX].
        """
        mask_data, mask_affine = load_volume(mask_path)
        raw_data, raw_affine = load_volume(raw_input_path)

        # 1. Binary check
        unique = set(np.unique(mask_data).tolist())
        if not unique.issubset({0.0, 1.0, 0, 1}):
            unexpected = unique - {0.0, 1.0, 0, 1}
            raise QualityGateError(
                f"Brain mask is not binary: unexpected values {sorted(unexpected)} "
                f"in {mask_path}"
            )

        # 2. Shape match
        if mask_data.shape != raw_data.shape:
            raise QualityGateError(
                f"Shape mismatch: mask {mask_data.shape} != raw input {raw_data.shape} "
                f"for {mask_path}"
            )

        # 3. Affine match
        if not np.allclose(mask_affine, raw_affine, atol=1e-4):
            max_delta = float(np.abs(mask_affine - raw_affine).max())
            raise QualityGateError(
                f"Affine mismatch (max_delta={max_delta:.6f}) between mask and "
                f"raw input for {mask_path}"
            )

        # 4. Foreground fraction
        foreground_frac = float(np.count_nonzero(mask_data)) / float(mask_data.size)
        if not (FOREGROUND_MIN <= foreground_frac <= FOREGROUND_MAX):
            raise QualityGateError(
                f"Foreground fraction {foreground_frac:.3f} outside "
                f"[{FOREGROUND_MIN}, {FOREGROUND_MAX}] in {mask_path} — "
                "degenerate mask (all-zero or all-one)"
            )

        logger.debug(
            "assert_structural passed: %s foreground=%.3f",
            mask_path.name, foreground_frac,
        )

    def check_dsc(
        self, predicted_mask_path: Path, reference_mask_path: Path
    ) -> float:
        """Phase 2: compute DSC between predicted brain mask and expert reference.

        Raises QualityGateError if DSC ≤ 0.85 or both masks are empty.
        Returns the DSC value on success.
        """
        pred = load_mask(predicted_mask_path).astype(bool)
        ref = load_mask(reference_mask_path).astype(bool)

        sum_pred = int(np.count_nonzero(pred))
        sum_ref = int(np.count_nonzero(ref))

        if sum_pred + sum_ref == 0:
            raise QualityGateError(
                f"Both masks are empty — cannot compute DSC for "
                f"{predicted_mask_path}"
            )

        intersection = int(np.count_nonzero(pred & ref))
        dsc = (2 * intersection) / (sum_pred + sum_ref)

        if dsc <= DSC_THRESHOLD:
            raise QualityGateError(
                f"DSC {dsc:.4f} ≤ {DSC_THRESHOLD} for {predicted_mask_path.name}"
            )

        logger.debug("check_dsc: %s dsc=%.4f", predicted_mask_path.name, dsc)
        return float(dsc)
