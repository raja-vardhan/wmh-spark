"""Unit coverage for sparse empty-brain summaries in scripts/run_pipeline.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from wmh_spark.io_utils import SubjectRecord, save_volume


def test_empty_brain_prediction_summary_matches_flair_geometry(tmp_path, synthetic_affine, synthetic_volume_shape):
    import scripts.run_pipeline as run_pipeline

    skull_root = tmp_path / "strip"
    subj_dir = skull_root / "TestSubj_amsterdam"
    subj_dir.mkdir(parents=True, exist_ok=True)
    affine = np.asarray(synthetic_affine, dtype=np.float32)
    shape = synthetic_volume_shape
    flair = np.random.default_rng(1).standard_normal(shape, dtype=np.float32)
    flair_path = subj_dir / "FLAIR_bet.nii.gz"
    save_volume(flair, affine, flair_path)
    prior = np.zeros(shape, dtype=np.float32)
    save_volume(prior, affine, subj_dir / "spatial_prior.nii.gz")

    record = SubjectRecord(
        subject_id="TestSubj_amsterdam",
        t1_path=str(tmp_path / "T1.nii.gz"),
        flair_path=str(tmp_path / "pre" / "FLAIR.nii.gz"),
        gt_mask_path=None,
    )
    summary = run_pipeline._distributed_prediction_result_for_empty_brain(skull_root, record)
    assert summary.shape == shape
    assert summary.raw_predicted_voxels == 0
    assert summary.max_wmh_probability is None
    assert summary.component_count == 0
    assert summary.positive_xyz == []
    pred = run_pipeline._prediction_mask_from_summary(summary)
    assert pred.shape == shape and int(pred.sum()) == 0
