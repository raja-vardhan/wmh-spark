"""End-to-end smoke: ``distributed_sparse`` engine through train → validation tuning → predict."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from wmh_spark.io_utils import save_volume


def _make_brain(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same synthetic brain anatomy as ``tests/conftest.py`` (small reproducible MRI)."""
    z, y, x = np.indices(shape, dtype=np.float32)
    cz, cy, cx = (s / 2 for s in shape)
    rz, ry, rx = (s / 2.5 for s in shape)
    ellipsoid = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    brain = ellipsoid < 1.0
    ventricles = ellipsoid < 0.05
    wm = brain & ~ventricles

    rng = np.random.default_rng(0)
    flair = np.zeros(shape, dtype=np.float32)
    flair[wm] = 100 + rng.normal(0, 5, wm.sum())
    flair[ventricles] = 20

    lesion = np.zeros(shape, dtype=bool)
    lz, ly, lx = (s // 2 - 3 for s in shape)
    lesion[lz : lz + 3, ly : ly + 4, lx : lx + 5] = True
    lesion &= wm
    flair[lesion] = 200

    t1 = np.zeros(shape, dtype=np.float32)
    t1[wm] = 150 + np.random.default_rng(1).normal(0, 5, wm.sum())
    t1[ventricles] = 30
    t1[lesion] = 152

    mask = lesion.astype(np.uint8)
    return flair, t1, mask


class _CopySkullStripper:
    """Structural HD-BET stand-in: copy raw stripped volumes + brain masks passing the smoke gate."""

    def __init__(self, output_dir: Path | str, **_kwargs) -> None:
        self.output_dir = Path(output_dir)
        self._run_id = "smoke-strip"

    def process_pair(self, pair):
        from wmh_spark.preprocessing.skull_strip import SkullStrippedOutput, _derive_stem

        subject_dir = self.output_dir / pair.subject_id
        if subject_dir.exists():
            shutil.rmtree(subject_dir)
        subject_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        t1_stem = _derive_stem(Path(pair.t1_scan.path))
        flair_stem = _derive_stem(Path(pair.flair_scan.path))

        t1_img = nib.load(str(pair.t1_scan.path))
        flair_img = nib.load(str(pair.flair_scan.path))
        t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
        flair = np.asarray(flair_img.dataobj, dtype=np.float32)
        affine = t1_img.affine

        t1_out = subject_dir / f"{t1_stem}_bet.nii.gz"
        t1_mask_out = subject_dir / f"{t1_stem}_bet_mask.nii.gz"
        flair_out = subject_dir / f"{flair_stem}_bet.nii.gz"
        flair_mask_out = subject_dir / f"{flair_stem}_bet_mask.nii.gz"

        save_volume(t1, affine, t1_out)
        save_volume(flair, affine, flair_out)

        bin_mask = ((np.abs(t1) > 1e-6) | (np.abs(flair) > 1e-6)).astype(np.uint8)
        save_volume(bin_mask, affine, t1_mask_out, dtype=np.uint8)
        save_volume(bin_mask, affine, flair_mask_out, dtype=np.uint8)

        return SkullStrippedOutput(
            subject_id=pair.subject_id,
            t1_stripped_path=t1_out,
            t1_mask_path=t1_mask_out,
            flair_stripped_path=flair_out,
            flair_mask_path=flair_mask_out,
            status="accepted",
            elapsed_seconds=time.time() - t0,
        )

    def _write_log(self, *_args: object, **_kwargs: object) -> None:
        return None


def _materialize_kaggle_tree(data_root: Path, affine: np.ndarray, shape: tuple[int, int, int]) -> tuple[list[str], list[str]]:
    """Write minimal Kaggle-style tree; returns ``(training_subject_ids, test_subject_ids)``."""

    def write_subject(split: str, site: str, scanner: str | None, folder: str) -> str:
        if scanner:
            subj_dir = Path(data_root) / split / site / scanner / folder
        else:
            subj_dir = Path(data_root) / split / site / folder
        pre = subj_dir / "pre"
        pre.mkdir(parents=True, exist_ok=True)
        flair, t1, mask = _make_brain(shape)
        save_volume(flair, affine, pre / "FLAIR.nii")
        save_volume(t1, affine, pre / "T1.nii")
        save_volume(mask.astype(np.uint8), affine, subj_dir / "wmh.nii", dtype=np.uint8)
        if scanner:
            scanner_clean = scanner.replace(".", "").replace(" ", "")
            return f"{site}_{scanner_clean}_{folder}"
        return f"{site}_{folder}"

    train_ids = [
        write_subject("training", "Amsterdam", "GE3T", "ta0"),
        write_subject("training", "Amsterdam", "GE3T", "ta1"),
        write_subject("training", "Singapore", None, "ts0"),
        write_subject("training", "Singapore", None, "ts1"),
    ]
    test_ids = [
        write_subject("test", "Amsterdam", "GE3T", "tea0"),
        write_subject("test", "Singapore", None, "tes0"),
    ]
    return train_ids, test_ids


@pytest.mark.integration
def test_distributed_sparse_pipeline_smoke(monkeypatch, tmp_path, synthetic_affine, synthetic_volume_shape):
    """Train + validation tuning + predicted masks for multi-subject distributed_sparse run."""
    import scripts.run_pipeline as run_pipeline

    monkeypatch.setattr(run_pipeline, "SkullStripper", _CopySkullStripper)

    affine = np.asarray(synthetic_affine, dtype=np.float32)
    shape = synthetic_volume_shape
    data_root = tmp_path / "datasets"
    out_root = tmp_path / "output"
    _, expected_test_ids = _materialize_kaggle_tree(data_root, affine, shape)
    assert set(expected_test_ids) == {"Amsterdam_GE3T_tea0", "Singapore_tes0"}

    argv_backup = sys.argv.copy()
    try:
        sys.argv = [
            "run_pipeline.py",
            "--data-root",
            str(data_root),
            "--layout",
            "kaggle-wmh",
            "--output-root",
            str(out_root),
            "--train-sites",
            "Amsterdam,Singapore",
            "--test-sites",
            "Amsterdam,Singapore",
            "--scanners",
            "GE3T",
            "--max-train",
            "4",
            "--max-test",
            "2",
            "--feature-set",
            "baseline",
            "--validation-fraction",
            "0.5",
            "--negative-sampling-ratio",
            "2.0",
            "--candidate-thresholds",
            "0.35,0.5",
            "--num-trees",
            "8",
            "--max-depth",
            "6",
            "--prediction-threshold",
            "0.5",
            "--pipeline-engine",
            "distributed_sparse",
            "--spark-master",
            "local[2]",
            "--spark-driver-memory",
            "2g",
            "--spark-shuffle-partitions",
            "8",
            "--training-partitions",
            "4",
            "--no-progress",
        ]
        assert run_pipeline.main() == 0
    finally:
        sys.argv = argv_backup

    summary = json.loads((out_root / "benchmarks" / "summary.json").read_text())
    assert summary["pipeline_engine"] == "distributed_sparse"
    assert summary["validation_threshold_selection"] is not None

    pred_root = out_root / "predictions"
    report_root = out_root / "per_subject"

    assert set(expected_test_ids) == {p.name for p in pred_root.iterdir() if p.is_dir()}

    for sid in expected_test_ids:
        pred_path = pred_root / sid / "wmh_pred.nii.gz"
        assert pred_path.exists(), sid
        stats_path = report_root / sid / "stats.json"
        payload = json.loads(stats_path.read_text())
        assert payload["subject_id"] == sid

