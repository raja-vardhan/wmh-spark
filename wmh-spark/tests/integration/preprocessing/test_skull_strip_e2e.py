"""Integration tests for skull_strip end-to-end pipeline.

These tests mock local HD-BET execution but exercise the full batch/single-subject
flow including output file layout, log writing, and overwrite behaviour.
T010, T024 per tasks.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest

from wmh_spark.preprocessing.skull_strip import (
    RawScan,
    ScanPair,
    SkullStripper,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_valid_nifti(path: Path, brain_fraction: float = 0.25) -> None:
    """Write a minimal NIfTI with a synthetic brain mask-like signal."""
    shape = (16, 16, 16)
    affine = np.eye(4, dtype=np.float32)
    data = np.zeros(shape, dtype=np.float32)
    # Fill brain region (~25% of voxels by default)
    r = int(shape[0] * brain_fraction ** (1 / 3))
    c = shape[0] // 2
    data[c - r : c + r, c - r : c + r, c - r : c + r] = 500.0
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _mock_hdbet_run(input_path: Path, output_path: Path) -> None:
    """Simulate HD-BET: write stripped volume + binary mask NIfTI."""
    shape = (16, 16, 16)
    affine = np.eye(4, dtype=np.float32)

    stripped = np.zeros(shape, dtype=np.float32)
    c = shape[0] // 2
    r = 3
    stripped[c - r : c + r, c - r : c + r, c - r : c + r] = 500.0

    mask = (stripped > 0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(stripped, affine), str(output_path))

    stem = output_path.name
    if stem.endswith(".nii.gz"):
        mask_name = stem[:-7] + "_mask.nii.gz"
    else:
        mask_name = stem[:-4] + "_mask.nii.gz"
    mask_path = output_path.parent / mask_name
    nib.save(nib.Nifti1Image(mask, affine), str(mask_path))


# ---------------------------------------------------------------------------
# T010 — smoke batch end-to-end
# ---------------------------------------------------------------------------

class TestSmokeBatch:
    def test_smoke_batch_produces_output_files(self, tmp_path):
        """Full batch in smoke-test mode: outputs exist, log written."""
        # Set up two synthetic scan pairs
        for sub_id in ["sub-001", "sub-002"]:
            sub_dir = tmp_path / "input" / sub_id
            sub_dir.mkdir(parents=True)
            _write_valid_nifti(sub_dir / "T1_RMS.nii.gz")
            _write_valid_nifti(sub_dir / "FLAIR.nii.gz")

        out_dir = tmp_path / "output"
        bench_path = tmp_path / "bench.json"

        stripper = SkullStripper(
            output_dir=out_dir,
            smoke_test=True,
            benchmark_out=bench_path,
        )

        with patch.object(stripper._runner, "run", side_effect=_mock_hdbet_run):
            results = stripper.process_batch(input_dir=tmp_path / "input")

        assert len(results) == 2

        for sub_id in ["sub-001", "sub-002"]:
            sub_out = out_dir / sub_id
            assert (sub_out / "T1_RMS_bet.nii.gz").exists(), f"Missing T1 bet for {sub_id}"
            assert (sub_out / "FLAIR_bet.nii.gz").exists(), f"Missing FLAIR bet for {sub_id}"
            assert (sub_out / "T1_RMS_bet_mask.nii.gz").exists(), f"Missing T1 mask for {sub_id}"
            assert (sub_out / "FLAIR_bet_mask.nii.gz").exists(), f"Missing FLAIR mask for {sub_id}"

        for r in results:
            assert r.status == "accepted", f"{r.subject_id} should be accepted"

    def test_smoke_batch_writes_benchmark_json(self, tmp_path):
        sub_dir = tmp_path / "input" / "sub-001"
        sub_dir.mkdir(parents=True)
        _write_valid_nifti(sub_dir / "T1_RMS.nii.gz")
        _write_valid_nifti(sub_dir / "FLAIR.nii.gz")

        bench_path = tmp_path / "bench.json"
        stripper = SkullStripper(
            output_dir=tmp_path / "output",
            smoke_test=True,
            benchmark_out=bench_path,
        )

        with patch.object(stripper._runner, "run", side_effect=_mock_hdbet_run):
            stripper.process_batch(input_dir=tmp_path / "input")

        assert bench_path.exists()
        bench = json.loads(bench_path.read_text())
        assert "subjects_per_hour" in bench, "Benchmark must include subjects_per_hour (Principle V)"
        assert "run_id" in bench
        assert bench["total_pairs"] == 1

    def test_single_pair_failure_does_not_halt_batch(self, tmp_path):
        """FR-009: an HD-BET error on one pair must not stop the rest."""
        for sub_id in ["sub-ok", "sub-fail"]:
            sub_dir = tmp_path / "input" / sub_id
            sub_dir.mkdir(parents=True)
            _write_valid_nifti(sub_dir / "T1_RMS.nii.gz")
            _write_valid_nifti(sub_dir / "FLAIR.nii.gz")

        from wmh_spark.preprocessing.skull_strip import HDBETExecutionError

        call_count = [0]

        def selective_fail(input_path: Path, output_path: Path) -> None:
            call_count[0] += 1
            if "sub-fail" in str(input_path):
                raise HDBETExecutionError("injected failure")
            _mock_hdbet_run(input_path, output_path)

        stripper = SkullStripper(
            output_dir=tmp_path / "output", smoke_test=True,
            benchmark_out=tmp_path / "bench.json",
        )
        with patch.object(stripper._runner, "run", side_effect=selective_fail):
            results = stripper.process_batch(input_dir=tmp_path / "input")

        assert len(results) == 2
        statuses = {r.subject_id: r.status for r in results}
        assert statuses["sub-ok"] == "accepted"
        assert statuses["sub-fail"] == "error"


# ---------------------------------------------------------------------------
# T024 — single-subject reprocessing
# ---------------------------------------------------------------------------

class TestSingleSubjectReprocess:
    def test_reprocess_restores_output_and_leaves_sibling_intact(self, tmp_path):
        """Reprocessing sub-001 updates its output; sub-002 output is untouched."""
        for sub_id in ["sub-001", "sub-002"]:
            sub_dir = tmp_path / "input" / sub_id
            sub_dir.mkdir(parents=True)
            _write_valid_nifti(sub_dir / "T1_RMS.nii.gz")
            _write_valid_nifti(sub_dir / "FLAIR.nii.gz")

        out_dir = tmp_path / "output"
        bench_path = tmp_path / "bench.json"

        # --- Full batch run ---
        stripper = SkullStripper(
            output_dir=out_dir, smoke_test=True, benchmark_out=bench_path
        )
        with patch.object(stripper._runner, "run", side_effect=_mock_hdbet_run):
            stripper.process_batch(input_dir=tmp_path / "input")

        sibling_bet = out_dir / "sub-002" / "T1_RMS_bet.nii.gz"
        assert sibling_bet.exists()
        sibling_mtime = sibling_bet.stat().st_mtime

        # Corrupt sub-001 output
        target_bet = out_dir / "sub-001" / "T1_RMS_bet.nii.gz"
        target_bet.write_bytes(b"corrupted")

        # --- Reprocess sub-001 only ---
        time.sleep(0.05)  # ensure mtime can differ
        stripper2 = SkullStripper(
            output_dir=out_dir, smoke_test=True, benchmark_out=bench_path
        )
        with patch.object(stripper2._runner, "run", side_effect=_mock_hdbet_run):
            stripper2.process_batch(
                input_dir=tmp_path / "input", subject="sub-001"
            )

        # sub-001 output restored (not b"corrupted")
        restored = (out_dir / "sub-001" / "T1_RMS_bet.nii.gz").read_bytes()
        assert restored != b"corrupted", "sub-001 output was not restored"

        # sub-002 output unchanged (same mtime)
        assert sibling_bet.stat().st_mtime == sibling_mtime, (
            "sub-002 output was modified — sibling should be untouched"
        )
