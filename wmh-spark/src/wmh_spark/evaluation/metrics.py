"""Distributed accuracy metrics for WMH segmentation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyspark.sql import functions as F

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for post-processed mask evaluation."""

    prediction_column: str = "postprocessed_mask"
    label_column: str = "label"
    dsc_threshold: float = 0.85


@dataclass(frozen=True)
class DiceResult:
    """Dice Similarity Coefficient result and supporting counts."""

    dsc: float
    threshold: float
    passed: bool
    predicted_positive: int
    reference_positive: int
    intersection: int

    def to_dict(self) -> dict:
        return {
            "dsc": self.dsc,
            "threshold": self.threshold,
            "passed": self.passed,
            "predicted_positive": self.predicted_positive,
            "reference_positive": self.reference_positive,
            "intersection": self.intersection,
        }


class AccuracyGateError(Exception):
    """Raised when reproduction accuracy does not satisfy the DSC gate."""


def _validate_columns(df: "DataFrame", config: EvaluationConfig) -> None:
    required = [config.prediction_column, config.label_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required evaluation columns: {missing}")


def calculate_dice_result(
    df: "DataFrame",
    config: EvaluationConfig | None = None,
) -> DiceResult:
    """Calculate DSC between post-processed predictions and expert labels."""
    config = config or EvaluationConfig()
    _validate_columns(df, config)

    pred = F.col(config.prediction_column).cast("int")
    label = F.col(config.label_column).cast("int")
    counts = df.agg(
        F.sum(F.when(pred != 0, F.lit(1)).otherwise(F.lit(0))).alias("predicted_positive"),
        F.sum(F.when(label != 0, F.lit(1)).otherwise(F.lit(0))).alias("reference_positive"),
        F.sum(F.when((pred != 0) & (label != 0), F.lit(1)).otherwise(F.lit(0))).alias(
            "intersection"
        ),
    ).first()

    predicted_positive = int(counts["predicted_positive"] or 0)
    reference_positive = int(counts["reference_positive"] or 0)
    intersection = int(counts["intersection"] or 0)
    denominator = predicted_positive + reference_positive
    if denominator == 0:
        raise ValueError("cannot calculate DSC when both masks are empty")

    dsc = (2.0 * intersection) / denominator
    return DiceResult(
        dsc=float(dsc),
        threshold=float(config.dsc_threshold),
        passed=bool(dsc > config.dsc_threshold),
        predicted_positive=predicted_positive,
        reference_positive=reference_positive,
        intersection=intersection,
    )


def assert_reproduction_accuracy(
    df: "DataFrame",
    config: EvaluationConfig | None = None,
) -> DiceResult:
    """Return DSC result if it satisfies `DSC > threshold`; otherwise raise."""
    config = config or EvaluationConfig()
    result = calculate_dice_result(df, config)
    if not result.passed:
        raise AccuracyGateError(
            f"DSC {result.dsc:.4f} must be > {result.threshold:.4f} "
            f"(predicted_positive={result.predicted_positive}, "
            f"reference_positive={result.reference_positive}, "
            f"intersection={result.intersection})"
        )
    return result
