"""k-NN baseline -- UBO Detector reproduction.

The original UBO Detector uses k-nearest-neighbors on a fixed feature space.
We reproduce this faithfully using scikit-learn's KNeighborsClassifier so
the comparison against Random Forest is on identical features and an
identical training set.

k-NN does not parallelize well in Spark (every prediction needs the full
training set), so we run it as a *single-node baseline* on the driver
after pooling the training samples. The trained model is then broadcast
to workers for inference. This is honest about k-NN's limitations and is
exactly the bottleneck the project is designed to remove.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from ..config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class KNNBundle:
    """Trained k-NN bundle. Picklable, ships cleanly via Spark broadcast."""

    model: KNeighborsClassifier
    n_train: int
    train_seconds: float


def train_knn(X: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> KNNBundle:
    """Fit k-NN on pooled training samples (driver-side)."""
    t0 = time.perf_counter()
    model = KNeighborsClassifier(
        n_neighbors=cfg.knn_k,
        weights="distance",
        n_jobs=-1,
        algorithm="auto",
    )
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    logger.info("k-NN trained on %d samples in %.1fs", X.shape[0], elapsed)
    return KNNBundle(model=model, n_train=X.shape[0], train_seconds=elapsed)


def predict_knn(bundle: KNNBundle, X: np.ndarray) -> np.ndarray:
    """Return foreground probabilities for each row of X."""
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    proba = bundle.model.predict_proba(X)
    # Some sklearn versions return only one column if y has a single class.
    if proba.shape[1] == 1:
        return np.zeros(X.shape[0], dtype=np.float32)
    return proba[:, 1].astype(np.float32)
