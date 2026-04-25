"""XGBoost on Spark.

The third point in our model comparison. XGBoost typically matches or beats
Random Forest on tabular data and is the most common modern alternative to
RF, so including it makes the ML evaluation publication-quality rather than
just illustrative.

We use the standalone XGBoost Python API rather than xgboost4j-spark to
keep the dependency tree light. Training is single-node (driver) but
multi-threaded; inference runs on workers via broadcast.

If the cohort outgrows driver memory, swap this for ``xgboost.spark`` which
ships a distributed implementation with the same API surface.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from ..config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class XGBBundle:
    model: object  # xgboost.XGBClassifier; not typed to avoid hard dep
    n_train: int
    train_seconds: float


def train_xgb_spark(X: np.ndarray, y: np.ndarray, cfg: ModelConfig) -> XGBBundle:
    """Fit an XGBoost classifier on pooled training samples.

    For very large cohorts switch to xgboost.spark.SparkXGBClassifier; the
    fitted estimator has the same predict_proba interface, so callers
    don't need to change.
    """
    from xgboost import XGBClassifier

    t0 = time.perf_counter()
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)

    model = XGBClassifier(
        n_estimators=cfg.xgb_n_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=neg / pos,  # explicit imbalance correction
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    logger.info("XGBoost trained on %d samples in %.1fs", X.shape[0], elapsed)
    return XGBBundle(model=model, n_train=X.shape[0], train_seconds=elapsed)


def predict_xgb_spark(bundle: XGBBundle, X: np.ndarray) -> np.ndarray:
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    proba = bundle.model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.zeros(X.shape[0], dtype=np.float32)
    return proba[:, 1].astype(np.float32)
