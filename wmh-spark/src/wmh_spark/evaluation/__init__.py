"""Evaluation metrics and benchmark logging APIs."""

from wmh_spark.evaluation.benchmarking import (
    SubjectBenchmarkRecord,
    ThroughputRecord,
    log_cluster_throughput,
    log_subject_benchmark,
    summarize_scaling,
)
from wmh_spark.evaluation.metrics import (
    AccuracyGateError,
    DiceResult,
    EvaluationConfig,
    assert_reproduction_accuracy,
    calculate_dice_result,
)

__all__ = [
    "AccuracyGateError",
    "DiceResult",
    "EvaluationConfig",
    "SubjectBenchmarkRecord",
    "ThroughputRecord",
    "assert_reproduction_accuracy",
    "calculate_dice_result",
    "log_cluster_throughput",
    "log_subject_benchmark",
    "summarize_scaling",
]
