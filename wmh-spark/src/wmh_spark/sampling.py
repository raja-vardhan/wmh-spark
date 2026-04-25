"""Class-balanced sampling for training set construction.

WMH voxels are typically <1% of the candidate set. Naive training trivially
achieves 99% accuracy by predicting all-negative. We address this by:

1. Subsampling negatives at a configurable ratio per subject.
2. Capping total voxels per subject so a few large brains don't dominate.
3. Pooling samples across the cohort into one Spark DataFrame for global
   training.

This is the only stage where we actually need a Spark DataFrame -- here it
genuinely earns its place because pooling samples across many subjects is
exactly what distributed shuffling is good at.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from .config import SamplingConfig
from .features import FEATURE_NAMES, FeatureBundle

logger = logging.getLogger(__name__)


def sample_subject(
    bundle: FeatureBundle,
    cfg: SamplingConfig,
    rng_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a class-balanced (X, y) sample from one subject's features.

    The seed is derived deterministically from the subject id + global seed
    so re-running the pipeline produces identical splits.
    """
    if not bundle.success or bundle.labels is None:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty(0, dtype=np.uint8)

    rng = np.random.default_rng(rng_seed)
    X, y = bundle.features, bundle.labels

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    if pos_idx.size == 0:
        # Subject has no lesions; sample a small slice of negatives so the
        # model still sees this subject's intensity distribution.
        n_neg = min(neg_idx.size, cfg.max_voxels_per_subject // 10)
        chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
        return X[chosen_neg], y[chosen_neg]

    # Cap positives first; negatives are derived from the ratio.
    n_pos = min(pos_idx.size, cfg.max_voxels_per_subject // (cfg.negative_to_positive_ratio + 1))
    n_neg = min(neg_idx.size, n_pos * cfg.negative_to_positive_ratio)

    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(keep)

    return X[keep], y[keep]


def to_spark_rows(
    samples: Iterator[tuple[str, np.ndarray, np.ndarray]],
):
    """Convert per-subject (X, y) arrays to (subject_id, features, label) rows.

    Yielding generator-style avoids materializing the full pooled training
    set on the driver. Spark's parallelize will partition naturally across
    workers.
    """
    for subject_id, X, y in samples:
        for row, label in zip(X, y):
            yield (subject_id, row.tolist(), int(label))
