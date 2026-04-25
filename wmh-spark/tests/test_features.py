"""Tests for feature extraction."""

from __future__ import annotations

import numpy as np

from wmh_spark.config import FeatureConfig
from wmh_spark.features import FEATURE_NAMES, extract_features


def test_extract_features_synthetic_shape(synthetic_subject):
    cfg = FeatureConfig()
    bundle = extract_features(
        subject_id="synth",
        flair_mni_path=str(synthetic_subject / "flair.nii.gz"),
        t1_mni_path=str(synthetic_subject / "t1.nii.gz"),
        brain_mask_mni_path=str(synthetic_subject / "brain_mask.nii.gz"),
        cfg=cfg,
        gt_mask_mni_path=str(synthetic_subject / "wmh_mask.nii.gz"),
    )
    assert bundle.success, bundle.error
    assert bundle.features.shape[1] == len(FEATURE_NAMES)
    assert bundle.features.shape[0] > 0
    assert bundle.features.shape[0] == bundle.voxel_indices.shape[0]
    assert bundle.labels is not None
    assert bundle.labels.shape == (bundle.features.shape[0],)


def test_features_finite(synthetic_subject):
    bundle = extract_features(
        subject_id="synth",
        flair_mni_path=str(synthetic_subject / "flair.nii.gz"),
        t1_mni_path=str(synthetic_subject / "t1.nii.gz"),
        brain_mask_mni_path=str(synthetic_subject / "brain_mask.nii.gz"),
        cfg=FeatureConfig(),
        gt_mask_mni_path=None,
    )
    # No NaNs, no infs -- downstream classifiers will choke on them.
    assert np.all(np.isfinite(bundle.features))


def test_lesion_voxels_have_higher_flair_z(synthetic_subject):
    """Sanity check: synthetic lesion should be flagged as high FLAIR z-score."""
    bundle = extract_features(
        subject_id="synth",
        flair_mni_path=str(synthetic_subject / "flair.nii.gz"),
        t1_mni_path=str(synthetic_subject / "t1.nii.gz"),
        brain_mask_mni_path=str(synthetic_subject / "brain_mask.nii.gz"),
        cfg=FeatureConfig(),
        gt_mask_mni_path=str(synthetic_subject / "wmh_mask.nii.gz"),
    )
    flair_z = bundle.features[:, 0]
    pos = bundle.labels == 1
    if pos.sum() > 0:
        # Lesion voxels should sit well above non-lesion voxels in FLAIR z-score.
        assert flair_z[pos].mean() > flair_z[~pos].mean()


def test_failure_returns_empty_bundle(tmp_path):
    """A bad path should produce a failed (but non-crashing) bundle."""
    bundle = extract_features(
        subject_id="missing",
        flair_mni_path=str(tmp_path / "nope.nii.gz"),
        t1_mni_path=str(tmp_path / "nope.nii.gz"),
        brain_mask_mni_path=str(tmp_path / "nope.nii.gz"),
        cfg=FeatureConfig(),
    )
    assert not bundle.success
    assert bundle.features.shape[0] == 0
    assert bundle.error is not None
