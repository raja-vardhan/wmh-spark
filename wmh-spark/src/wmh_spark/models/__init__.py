"""Classifiers benchmarked in this project.

We deliberately implement three classifiers with identical input features:
- k-NN baseline (UBO Detector reproduction)
- Spark MLlib Random Forest (the proposed replacement)
- XGBoost on Spark (a stronger alternative point)

Each module exposes the same two functions:
    train(spark, train_df, cfg) -> fitted_model
    predict(model, features_2d_ndarray) -> probabilities (n,) float32
"""

from .knn_baseline import train_knn, predict_knn
from .rf_spark import train_rf_spark, predict_rf_spark
from .xgb_spark import train_xgb_spark, predict_xgb_spark

__all__ = [
    "train_knn", "predict_knn",
    "train_rf_spark", "predict_rf_spark",
    "train_xgb_spark", "predict_xgb_spark",
]
