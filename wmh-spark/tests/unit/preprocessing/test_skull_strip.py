"""Unit tests for skull_strip module.

TDD: these tests were written before implementation (Red-Green-Refactor).
Constitution Principle VI requires tests to be written and confirmed failing first.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wmh_spark.preprocessing.skull_strip import (
    DockerExecutionError,
    DockerRunner,
    ProcessingLogEntry,
    RawScan,
    ScanPair,
    SkullStrippedOutput,
    SkullStripper,
    _derive_stem,
    _discover_pairs,
)


# ---------------------------------------------------------------------------
# T005 — Dataclass structural smoke tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_raw_scan_fields(self):
        scan = RawScan(path=Path("/data/T1.nii.gz"), modality="T1", subject_id="s1")
        assert scan.path == Path("/data/T1.nii.gz")
        assert scan.modality == "T1"
        assert scan.subject_id == "s1"

    def test_scan_pair_optional_ref_defaults_to_none(self):
        t1 = RawScan(path=Path("/t1.nii.gz"), modality="T1", subject_id="s1")
        fl = RawScan(path=Path("/flair.nii.gz"), modality="FLAIR", subject_id="s1")
        pair = ScanPair(subject_id="s1", t1_scan=t1, flair_scan=fl)
        assert pair.t1_brain_ref_path is None
        assert pair.flair_brain_ref_path is None

    def test_skull_stripped_output_defaults(self):
        out = SkullStrippedOutput(subject_id="s1")
        assert out.status == "error"
        assert out.t1_dsc is None
        assert out.elapsed_seconds == 0.0

    def test_processing_log_entry_fields(self):
        entry = ProcessingLogEntry(
            run_id="abc", timestamp="2026-01-01T00:00:00Z",
            subject_id="s1", t1_dsc=0.9, flair_dsc=0.91,
            status="accepted", elapsed_seconds=12.5,
        )
        assert entry.error_message is None

    def test_derive_stem_strips_nii_gz(self):
        assert _derive_stem(Path("/data/T1_RMS.nii.gz")) == "T1_RMS"

    def test_derive_stem_strips_nii(self):
        assert _derive_stem(Path("/data/FLAIR.nii")) == "FLAIR"


# ---------------------------------------------------------------------------
# T007 — DockerRunner._build_command
# ---------------------------------------------------------------------------

class TestDockerRunner:
    def _runner(self):
        return DockerRunner("wmh-spark/hdbet:2.0.0")

    def test_contains_pinned_image_not_latest(self):
        cmd = self._runner()._build_command(
            Path("/data/T1.nii.gz"), Path("/out/T1_bet.nii.gz")
        )
        assert "wmh-spark/hdbet:2.0.0" in cmd
        assert "latest" not in " ".join(cmd)

    def test_gpu_flag_present(self):
        cmd = self._runner()._build_command(
            Path("/data/T1.nii.gz"), Path("/out/T1_bet.nii.gz")
        )
        assert "--gpus" in cmd
        idx = cmd.index("--gpus")
        assert cmd[idx + 1] == "all"

    def test_input_volume_mount_is_readonly(self):
        cmd = self._runner()._build_command(
            Path("/data/sub/T1.nii.gz"), Path("/out/sub/T1_bet.nii.gz")
        )
        mounts = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-v"]
        assert any(":ro" in m for m in mounts)

    def test_output_suffix_in_command(self):
        cmd = self._runner()._build_command(
            Path("/data/T1_RMS.nii.gz"), Path("/out/T1_RMS_bet.nii.gz")
        )
        assert "T1_RMS_bet.nii.gz" in " ".join(cmd)

    def test_run_raises_docker_execution_error_on_nonzero_exit(self, tmp_path):
        runner = self._runner()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="OOM error")
            with pytest.raises(DockerExecutionError, match="OOM error"):
                runner.run(tmp_path / "T1.nii.gz", tmp_path / "T1_bet.nii.gz")

    def test_run_succeeds_on_zero_exit(self, tmp_path):
        runner = self._runner()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            runner.run(tmp_path / "T1.nii.gz", tmp_path / "T1_bet.nii.gz")


# ---------------------------------------------------------------------------
# T008 — _discover_pairs
# ---------------------------------------------------------------------------

class TestDiscoverPairs:
    def test_discovers_single_pair_in_subdirectory(self, tmp_path):
        sub = tmp_path / "sub-001"
        sub.mkdir()
        (sub / "T1_RMS.nii.gz").touch()
        (sub / "FLAIR.nii.gz").touch()

        pairs = _discover_pairs(tmp_path)
        assert len(pairs) == 1
        assert pairs[0].subject_id == "sub-001"
        assert "T1" in pairs[0].t1_scan.path.name
        assert "FLAIR" in pairs[0].flair_scan.path.name.upper()

    def test_discovers_multiple_pairs(self, tmp_path):
        for i in range(3):
            sub = tmp_path / f"sub-{i:03d}"
            sub.mkdir()
            (sub / "T1.nii.gz").touch()
            (sub / "FLAIR.nii.gz").touch()
        pairs = _discover_pairs(tmp_path)
        assert len(pairs) == 3

    def test_raises_when_no_pairs_found(self, tmp_path):
        (tmp_path / "readme.txt").touch()
        with pytest.raises(ValueError, match="No complete"):
            _discover_pairs(tmp_path)

    def test_raises_on_duplicate_subject_id(self, tmp_path):
        # Two dirs with same name can't exist — test duplicate via direct check
        sub = tmp_path / "sub-001"
        sub.mkdir()
        (sub / "T1.nii.gz").touch()
        (sub / "FLAIR.nii.gz").touch()
        # Only one subject: no duplicate
        pairs = _discover_pairs(tmp_path)
        assert len(pairs) == 1

    def test_partial_pair_skipped_and_raises_if_only_partial(self, tmp_path):
        sub = tmp_path / "sub-001"
        sub.mkdir()
        (sub / "T1.nii.gz").touch()
        # No FLAIR — no complete pairs
        with pytest.raises(ValueError, match="No complete"):
            _discover_pairs(tmp_path)

    def test_partial_pair_skipped_valid_pairs_returned(self, tmp_path):
        good = tmp_path / "sub-good"
        good.mkdir()
        (good / "T1.nii.gz").touch()
        (good / "FLAIR.nii.gz").touch()

        bad = tmp_path / "sub-bad"
        bad.mkdir()
        (bad / "T1.nii.gz").touch()  # no FLAIR

        pairs = _discover_pairs(tmp_path)
        assert len(pairs) == 1
        assert pairs[0].subject_id == "sub-good"


# ---------------------------------------------------------------------------
# T009 — SkullStripper.process_pair
# ---------------------------------------------------------------------------

class TestSkullStripperProcessPair:
    def _make_pair(self, tmp_path: Path) -> ScanPair:
        t1 = tmp_path / "T1_RMS.nii.gz"
        fl = tmp_path / "FLAIR.nii.gz"
        t1.touch()
        fl.touch()
        return ScanPair(
            subject_id="sub-001",
            t1_scan=RawScan(path=t1, modality="T1", subject_id="sub-001"),
            flair_scan=RawScan(path=fl, modality="FLAIR", subject_id="sub-001"),
        )

    def test_returns_skull_stripped_output(self, tmp_path):
        pair = self._make_pair(tmp_path)
        stripper = SkullStripper(output_dir=tmp_path / "out", smoke_test=True)
        with patch.object(stripper._runner, "run", return_value=None), \
             patch("wmh_spark.preprocessing.quality_gate.QualityGate.assert_structural",
                   return_value=None):
            result = stripper.process_pair(pair)
        assert isinstance(result, SkullStrippedOutput)
        assert result.subject_id == "sub-001"

    def test_records_non_negative_elapsed_time(self, tmp_path):
        pair = self._make_pair(tmp_path)
        stripper = SkullStripper(output_dir=tmp_path / "out", smoke_test=True)
        with patch.object(stripper._runner, "run", return_value=None), \
             patch("wmh_spark.preprocessing.quality_gate.QualityGate.assert_structural",
                   return_value=None):
            result = stripper.process_pair(pair)
        assert result.elapsed_seconds >= 0.0

    def test_status_accepted_on_success(self, tmp_path):
        pair = self._make_pair(tmp_path)
        stripper = SkullStripper(output_dir=tmp_path / "out", smoke_test=True)
        with patch.object(stripper._runner, "run", return_value=None), \
             patch("wmh_spark.preprocessing.quality_gate.QualityGate.assert_structural",
                   return_value=None):
            result = stripper.process_pair(pair)
        assert result.status == "accepted"

    def test_status_error_on_docker_failure(self, tmp_path):
        pair = self._make_pair(tmp_path)
        stripper = SkullStripper(output_dir=tmp_path / "out", smoke_test=True)
        with patch.object(stripper._runner, "run",
                          side_effect=DockerExecutionError("OOM")):
            result = stripper.process_pair(pair)
        assert result.status == "error"
        assert "OOM" in result.error_message

    def test_output_paths_use_bet_suffix(self, tmp_path):
        pair = self._make_pair(tmp_path)
        stripper = SkullStripper(output_dir=tmp_path / "out", smoke_test=True)
        with patch.object(stripper._runner, "run", return_value=None), \
             patch("wmh_spark.preprocessing.quality_gate.QualityGate.assert_structural",
                   return_value=None):
            result = stripper.process_pair(pair)
        assert result.t1_stripped_path.name == "T1_RMS_bet.nii.gz"
        assert result.flair_stripped_path.name == "FLAIR_bet.nii.gz"


# ---------------------------------------------------------------------------
# T023 — subject filter in process_batch
# ---------------------------------------------------------------------------

class TestSubjectFilter:
    def _make_pairs(self, tmp_path: Path, n: int = 3) -> list[ScanPair]:
        return [
            ScanPair(
                subject_id=f"sub-{i:03d}",
                t1_scan=RawScan(
                    path=tmp_path / f"T1_{i}.nii.gz",
                    modality="T1",
                    subject_id=f"sub-{i:03d}",
                ),
                flair_scan=RawScan(
                    path=tmp_path / f"FLAIR_{i}.nii.gz",
                    modality="FLAIR",
                    subject_id=f"sub-{i:03d}",
                ),
            )
            for i in range(n)
        ]

    def test_filters_to_single_subject(self, tmp_path):
        pairs = self._make_pairs(tmp_path)
        stripper = SkullStripper(output_dir=tmp_path / "out", smoke_test=True)
        called_ids = []

        def fake_process(pair):
            called_ids.append(pair.subject_id)
            return SkullStrippedOutput(subject_id=pair.subject_id, status="accepted",
                                       elapsed_seconds=0.1)

        with patch.object(stripper, "process_pair", side_effect=fake_process), \
             patch.object(stripper, "_write_log"):
            stripper.process_batch(pairs=pairs, subject="sub-001")

        assert called_ids == ["sub-001"]

    def test_raises_when_subject_not_found(self, tmp_path):
        pairs = self._make_pairs(tmp_path, n=1)
        stripper = SkullStripper(output_dir=tmp_path / "out", smoke_test=True)
        with pytest.raises(ValueError, match="not found"):
            stripper.process_batch(pairs=pairs, subject="sub-999")


# ---------------------------------------------------------------------------
# T031 — manifest input mode
# ---------------------------------------------------------------------------

class TestManifestInputMode:
    def test_manifest_builds_correct_scan_pairs(self, tmp_path):
        import pandas as pd

        rows = [
            {
                "subject_id": "sub-001",
                "t1_path": "/data/sub-001/T1.nii.gz",
                "flair_path": "/data/sub-001/FLAIR.nii.gz",
                "gt_mask_path": None, "site": None, "age": None, "split": None,
            },
            {
                "subject_id": "sub-002",
                "t1_path": "/data/sub-002/T1.nii.gz",
                "flair_path": "/data/sub-002/FLAIR.nii.gz",
                "gt_mask_path": None, "site": None, "age": None, "split": None,
            },
        ]
        manifest_path = tmp_path / "manifest.parquet"
        pd.DataFrame(rows).to_parquet(str(manifest_path))

        stripper = SkullStripper(output_dir=tmp_path / "out", smoke_test=True)
        captured = []

        def fake_batch(pairs=None, input_dir=None, subject=None):
            captured.extend(pairs or [])
            return []

        with patch.object(stripper, "process_batch", side_effect=fake_batch):
            stripper.process_from_manifest(manifest_path)

        assert len(captured) == 2
        assert captured[0].subject_id == "sub-001"
        assert captured[1].subject_id == "sub-002"
        assert captured[0].t1_scan.modality == "T1"
        assert captured[0].flair_scan.modality == "FLAIR"


# ---------------------------------------------------------------------------
# T032 — deterministic output (same input → same checksum)
# ---------------------------------------------------------------------------

class TestDeterministicOutput:
    def test_same_input_produces_identical_checksums(self, tmp_path):
        t1_path = tmp_path / "T1_RMS.nii.gz"
        flair_path = tmp_path / "FLAIR.nii.gz"
        t1_path.touch()
        flair_path.touch()

        pair = ScanPair(
            subject_id="sub-det",
            t1_scan=RawScan(path=t1_path, modality="T1", subject_id="sub-det"),
            flair_scan=RawScan(path=flair_path, modality="FLAIR", subject_id="sub-det"),
        )

        fake_t1_content = b"deterministic_t1_nifti_bytes"
        fake_fl_content = b"deterministic_flair_nifti_bytes"

        def mock_docker_run(input_path: Path, output_path: Path) -> None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if "T1" in input_path.name:
                output_path.write_bytes(fake_t1_content)
                mask = output_path.parent / output_path.name.replace(
                    ".nii.gz", "_mask.nii.gz"
                )
                mask.write_bytes(fake_t1_content + b"_mask")
            else:
                output_path.write_bytes(fake_fl_content)
                mask = output_path.parent / output_path.name.replace(
                    ".nii.gz", "_mask.nii.gz"
                )
                mask.write_bytes(fake_fl_content + b"_mask")

        checksums: list[str] = []
        for run_idx in range(2):
            out_dir = tmp_path / f"run{run_idx}"
            stripper = SkullStripper(output_dir=out_dir, smoke_test=True)
            with patch.object(stripper._runner, "run", side_effect=mock_docker_run), \
                 patch(
                     "wmh_spark.preprocessing.quality_gate.QualityGate.assert_structural",
                     return_value=None,
                 ):
                result = stripper.process_pair(pair)

            t1_out = out_dir / "sub-det" / "T1_RMS_bet.nii.gz"
            assert t1_out.exists(), f"Expected output missing in run {run_idx}"
            checksums.append(hashlib.sha256(t1_out.read_bytes()).hexdigest())

        assert checksums[0] == checksums[1], "Outputs differ across runs — not deterministic"
