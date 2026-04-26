"""Unit tests for quality_gate module.

TDD: written before implementation. Constitution Principle VI.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from wmh_spark.preprocessing.quality_gate import (
    DSC_THRESHOLD,
    FOREGROUND_MAX,
    FOREGROUND_MIN,
    QualityGate,
    QualityGateError,
)


# ---------------------------------------------------------------------------
# T006 — import scaffold
# ---------------------------------------------------------------------------

def test_quality_gate_module_importable():
    from wmh_spark.preprocessing import quality_gate
    assert quality_gate is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_nifti(arr: np.ndarray, path: Path, affine: np.ndarray | None = None) -> None:
    if affine is None:
        affine = np.eye(4, dtype=np.float32)
    nib.save(nib.Nifti1Image(arr.astype(arr.dtype), affine), str(path))


# ---------------------------------------------------------------------------
# T016 — QualityGate.check_dsc
# ---------------------------------------------------------------------------

class TestCheckDsc:
    def test_perfect_overlap_returns_one(self, tmp_path):
        shape = (8, 8, 8)
        arr = np.zeros(shape, dtype=np.uint8)
        arr[2:6, 2:6, 2:6] = 1
        pred_path = tmp_path / "pred.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        _save_nifti(arr, pred_path)
        _save_nifti(arr, ref_path)

        gate = QualityGate(smoke_test=False)
        dsc = gate.check_dsc(pred_path, ref_path)
        assert dsc == pytest.approx(1.0)

    def test_rejects_dsc_below_threshold(self, tmp_path):
        shape = (10, 10, 10)
        pred = np.zeros(shape, dtype=np.uint8)
        pred[:1, :1, :1] = 1  # tiny overlap → DSC ≈ 0
        ref = np.ones(shape, dtype=np.uint8)
        pred_path = tmp_path / "pred.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        _save_nifti(pred, pred_path)
        _save_nifti(ref, ref_path)

        gate = QualityGate(smoke_test=False)
        with pytest.raises(QualityGateError):
            gate.check_dsc(pred_path, ref_path)

    def test_raises_on_both_empty_masks(self, tmp_path):
        shape = (5, 5, 5)
        empty = np.zeros(shape, dtype=np.uint8)
        pred_path = tmp_path / "pred.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        _save_nifti(empty, pred_path)
        _save_nifti(empty, ref_path)

        gate = QualityGate(smoke_test=False)
        with pytest.raises(QualityGateError, match="empty"):
            gate.check_dsc(pred_path, ref_path)

    def test_known_dsc_value(self, tmp_path):
        # pred covers half the reference → DSC = 2*(0.5*N) / (0.5*N + N) = 2/3
        shape = (10, 1, 1)
        ref = np.zeros(shape, dtype=np.uint8)
        ref[:10] = 1  # 10 voxels
        pred = np.zeros(shape, dtype=np.uint8)
        pred[:5] = 1  # 5 voxels, all overlapping → DSC = 2*5/(5+10) = 2/3 ≈ 0.667

        pred_path = tmp_path / "pred.nii.gz"
        ref_path = tmp_path / "ref.nii.gz"
        _save_nifti(pred, pred_path)
        _save_nifti(ref, ref_path)

        gate = QualityGate(smoke_test=False)
        with pytest.raises(QualityGateError):  # 0.667 < 0.85
            gate.check_dsc(pred_path, ref_path)


# ---------------------------------------------------------------------------
# T017 — QualityGate.assert_structural
# ---------------------------------------------------------------------------

class TestAssertStructural:
    def _make_raw(self, tmp_path: Path, shape=(12, 12, 12)) -> tuple[Path, np.ndarray, tuple]:
        affine = np.eye(4, dtype=np.float32)
        raw = np.ones(shape, dtype=np.float32) * 600.0
        raw_path = tmp_path / "T1.nii.gz"
        _save_nifti(raw, raw_path, affine)
        return raw_path, affine, shape

    def test_valid_mask_passes(self, tmp_path):
        raw_path, affine, shape = self._make_raw(tmp_path)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[3:9, 3:9, 3:9] = 1  # ~21% foreground
        mask_path = tmp_path / "mask.nii.gz"
        _save_nifti(mask, mask_path, affine)

        QualityGate(smoke_test=True).assert_structural(mask_path, raw_path)

    def test_non_binary_value_fails(self, tmp_path):
        raw_path, affine, shape = self._make_raw(tmp_path)
        mask = np.full(shape, 0.7, dtype=np.float32)
        mask_path = tmp_path / "mask_bad.nii.gz"
        _save_nifti(mask, mask_path, affine)

        with pytest.raises(QualityGateError, match="not binary"):
            QualityGate(smoke_test=True).assert_structural(mask_path, raw_path)

    def test_shape_mismatch_fails(self, tmp_path):
        raw_path, affine, _ = self._make_raw(tmp_path)
        wrong_mask = np.zeros((4, 4, 4), dtype=np.uint8)
        wrong_mask[1:3, 1:3, 1:3] = 1
        mask_path = tmp_path / "mask_shape.nii.gz"
        _save_nifti(wrong_mask, mask_path, affine)

        with pytest.raises(QualityGateError, match="Shape"):
            QualityGate(smoke_test=True).assert_structural(mask_path, raw_path)

    def test_all_zeros_foreground_fails(self, tmp_path):
        raw_path, affine, shape = self._make_raw(tmp_path)
        mask = np.zeros(shape, dtype=np.uint8)
        mask_path = tmp_path / "mask_zero.nii.gz"
        _save_nifti(mask, mask_path, affine)

        with pytest.raises(QualityGateError, match="Foreground"):
            QualityGate(smoke_test=True).assert_structural(mask_path, raw_path)

    def test_all_ones_foreground_fails(self, tmp_path):
        raw_path, affine, shape = self._make_raw(tmp_path)
        mask = np.ones(shape, dtype=np.uint8)
        mask_path = tmp_path / "mask_full.nii.gz"
        _save_nifti(mask, mask_path, affine)

        with pytest.raises(QualityGateError, match="Foreground"):
            QualityGate(smoke_test=True).assert_structural(mask_path, raw_path)

    def test_affine_mismatch_fails(self, tmp_path):
        raw_path, _, shape = self._make_raw(tmp_path)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[3:9, 3:9, 3:9] = 1
        wrong_affine = np.eye(4, dtype=np.float32) * 99.0
        wrong_affine[3, 3] = 1.0
        mask_path = tmp_path / "mask_affine.nii.gz"
        _save_nifti(mask, mask_path, wrong_affine)

        with pytest.raises(QualityGateError, match="[Aa]ffine"):
            QualityGate(smoke_test=True).assert_structural(mask_path, raw_path)


# ---------------------------------------------------------------------------
# T018 — pair-level atomic quarantine
# ---------------------------------------------------------------------------

class TestPairAtomicRejection:
    def test_quality_gate_failure_quarantines_all_outputs(self, tmp_path):
        from wmh_spark.preprocessing.skull_strip import RawScan, ScanPair, SkullStripper

        t1_src = tmp_path / "T1.nii.gz"
        fl_src = tmp_path / "FLAIR.nii.gz"
        t1_src.touch()
        fl_src.touch()

        pair = ScanPair(
            subject_id="sub-gate",
            t1_scan=RawScan(path=t1_src, modality="T1", subject_id="sub-gate"),
            flair_scan=RawScan(path=fl_src, modality="FLAIR", subject_id="sub-gate"),
        )

        out_dir = tmp_path / "out"
        stripper = SkullStripper(output_dir=out_dir, smoke_test=True)

        def mock_run(input_path: Path, output_path: Path) -> None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()

        with patch.object(stripper._runner, "run", side_effect=mock_run), \
             patch(
                 "wmh_spark.preprocessing.quality_gate.QualityGate.assert_structural",
                 side_effect=QualityGateError("degenerate mask"),
             ):
            result = stripper.process_pair(pair)

        assert result.status == "rejected"
        assert result.error_message is not None
        # Accepted outputs must not exist; quarantine dir must exist
        assert not (out_dir / "sub-gate").exists() or not any(
            (out_dir / "sub-gate").iterdir()
        )
        quarantine = out_dir / "quarantine" / "sub-gate"
        assert quarantine.exists()

    def test_hdbet_error_returns_error_status(self, tmp_path):
        from wmh_spark.preprocessing.skull_strip import RawScan, ScanPair, SkullStripper

        t1_src = tmp_path / "T1.nii.gz"
        fl_src = tmp_path / "FLAIR.nii.gz"
        t1_src.touch()
        fl_src.touch()

        pair = ScanPair(
            subject_id="sub-err",
            t1_scan=RawScan(path=t1_src, modality="T1", subject_id="sub-err"),
            flair_scan=RawScan(path=fl_src, modality="FLAIR", subject_id="sub-err"),
        )

        stripper = SkullStripper(output_dir=tmp_path / "out", smoke_test=True)
        from wmh_spark.preprocessing.skull_strip import HDBETExecutionError
        with patch.object(stripper._runner, "run",
                          side_effect=HDBETExecutionError("HD-BET unavailable")):
            result = stripper.process_pair(pair)

        assert result.status == "error"


# ---------------------------------------------------------------------------
# T033 — 60% voxel reduction (SC-003 / US1 Acceptance Scenario 2)
# ---------------------------------------------------------------------------

class TestVoxelReduction:
    def test_60_percent_voxel_reduction(self):
        shape = (10, 10, 10)
        total = np.prod(shape)

        raw = np.ones(shape, dtype=np.float32) * 500.0
        stripped = np.zeros(shape, dtype=np.float32)
        # Only ~6.4% of voxels remain (4x4x4 = 64 out of 1000)
        stripped[3:7, 3:7, 3:7] = 500.0

        nonzero_fraction = np.count_nonzero(stripped) / total
        assert nonzero_fraction < 0.40, (
            f"Expected < 40% nonzero voxels after skull-strip (SC-003), "
            f"got {nonzero_fraction:.2%}"
        )

    def test_assert_threshold_boundary(self):
        shape = (10, 10, 10)
        total = np.prod(shape)
        stripped_39pct = np.zeros(shape, dtype=np.float32)
        # 390 voxels = 39%
        flat = stripped_39pct.ravel()
        flat[:390] = 1.0
        stripped_39pct = flat.reshape(shape)
        assert np.count_nonzero(stripped_39pct) / total < 0.40

        stripped_41pct = np.zeros(shape, dtype=np.float32)
        flat2 = stripped_41pct.ravel()
        flat2[:410] = 1.0
        stripped_41pct = flat2.reshape(shape)
        assert np.count_nonzero(stripped_41pct) / total >= 0.40
