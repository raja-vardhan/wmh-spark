"""Tests for partial skull-strip cache reuse in scripts/run_pipeline.py."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from wmh_spark.io_utils import SubjectRecord


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-nifti")


class _FakeSkullStripper:
    def __init__(self, output_dir: Path, **_kwargs) -> None:
        self.output_dir = Path(output_dir)
        self._run_id = "fake-run"

    def process_pair(self, pair):  # matches the real API shape closely enough
        subj = self.output_dir / pair.subject_id
        subj.mkdir(parents=True, exist_ok=True)
        # The pipeline expects these exact filenames derived from the raw inputs.
        _touch(subj / "T1_bet.nii.gz")
        _touch(subj / "FLAIR_bet.nii.gz")
        _touch(subj / "FLAIR_bet_mask.nii.gz")
        return Namespace(
            subject_id=pair.subject_id,
            status="accepted",
            t1_dsc=None,
            flair_dsc=None,
            elapsed_seconds=0.1,
            error_message=None,
        )

    def _write_log(self, *_args, **_kwargs) -> None:
        return None


def test_stage_skull_strip_reuses_cached_subjects(tmp_path, monkeypatch):
    import scripts.run_pipeline as run_pipeline

    monkeypatch.setattr(run_pipeline, "SkullStripper", _FakeSkullStripper)

    cached_root = tmp_path / "cached_skull_stripped"
    output_root = tmp_path / "run_outputs"

    cached_subject = SubjectRecord(
        subject_id="cached_subj",
        t1_path="/fake/raw/T1.nii.gz",
        flair_path="/fake/raw/FLAIR.nii.gz",
        gt_mask_path=None,
    )
    missing_subject = SubjectRecord(
        subject_id="missing_subj",
        t1_path="/fake/raw/T1.nii.gz",
        flair_path="/fake/raw/FLAIR.nii.gz",
        gt_mask_path=None,
    )

    cached_subj_dir = cached_root / cached_subject.subject_id
    _touch(cached_subj_dir / "T1_bet.nii.gz")
    _touch(cached_subj_dir / "FLAIR_bet.nii.gz")
    _touch(cached_subj_dir / "FLAIR_bet_mask.nii.gz")

    args = Namespace(
        output_root=output_root,
        skip_skull_strip=False,
        skull_strip_root=cached_root,
        hdbet_device="cpu",
        enable_tta=False,
        no_progress=True,
    )

    skull_root = run_pipeline.stage_skull_strip(
        args,
        train_records=[cached_subject],
        test_records=[missing_subject],
    )

    cached_out_dir = skull_root / cached_subject.subject_id
    assert (cached_out_dir / "T1_bet.nii.gz").is_symlink()
    assert (cached_out_dir / "FLAIR_bet.nii.gz").is_symlink()
    assert (cached_out_dir / "FLAIR_bet_mask.nii.gz").is_symlink()

    missing_out_dir = skull_root / missing_subject.subject_id
    assert (missing_out_dir / "T1_bet.nii.gz").exists()
    assert (missing_out_dir / "FLAIR_bet.nii.gz").exists()
    assert (missing_out_dir / "FLAIR_bet_mask.nii.gz").exists()

