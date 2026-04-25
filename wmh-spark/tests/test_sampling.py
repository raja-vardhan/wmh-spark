"""Tests for class-balanced sampling."""

from __future__ import annotations

import numpy as np

from wmh_spark.config import SamplingConfig
from wmh_spark.features import FEATURE_NAMES, FeatureBundle
from wmh_spark.sampling import sample_subject


def _bundle_with(n_pos: int, n_neg: int, subject_id: str = "s1"):
    n = n_pos + n_neg
    feats = np.random.default_rng(0).normal(size=(n, len(FEATURE_NAMES))).astype(np.float32)
    labels = np.concatenate([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)])
    return FeatureBundle(
        subject_id=subject_id,
        features=feats,
        voxel_indices=np.arange(n, dtype=np.int64),
        volume_shape=(1, 1, n),
        affine=np.eye(4),
        labels=labels,
        success=True,
    )


def test_sampling_respects_ratio():
    bundle = _bundle_with(n_pos=100, n_neg=10_000)
    cfg = SamplingConfig(negative_to_positive_ratio=5, max_voxels_per_subject=10_000)
    X, y = sample_subject(bundle, cfg, rng_seed=42)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    assert n_pos > 0
    # Negatives shouldn't exceed ratio * positives.
    assert n_neg <= cfg.negative_to_positive_ratio * n_pos
    assert X.shape[0] == n_pos + n_neg


def test_sampling_caps_voxels_per_subject():
    bundle = _bundle_with(n_pos=10_000, n_neg=100_000)
    cfg = SamplingConfig(negative_to_positive_ratio=5, max_voxels_per_subject=600)
    X, y = sample_subject(bundle, cfg, rng_seed=42)
    # Cap is approximate (positives capped first, then negatives derived).
    assert X.shape[0] <= cfg.max_voxels_per_subject + 5


def test_sampling_handles_no_positives():
    bundle = _bundle_with(n_pos=0, n_neg=5000)
    cfg = SamplingConfig()
    X, y = sample_subject(bundle, cfg, rng_seed=42)
    # Should still return some negatives so the model sees this subject's
    # intensity distribution at training time.
    assert X.shape[0] > 0
    assert (y == 1).sum() == 0


def test_sampling_deterministic_for_same_seed():
    bundle = _bundle_with(n_pos=500, n_neg=10_000)
    cfg = SamplingConfig(negative_to_positive_ratio=5, max_voxels_per_subject=5000)
    X1, y1 = sample_subject(bundle, cfg, rng_seed=42)
    X2, y2 = sample_subject(bundle, cfg, rng_seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_sampling_failed_bundle_returns_empty():
    bundle = FeatureBundle(
        subject_id="bad",
        features=np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
        voxel_indices=np.empty(0, dtype=np.int64),
        volume_shape=(0, 0, 0),
        affine=np.eye(4),
        labels=None,
        success=False,
    )
    X, y = sample_subject(bundle, SamplingConfig(), rng_seed=0)
    assert X.shape[0] == 0
    assert y.shape[0] == 0
