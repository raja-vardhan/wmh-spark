"""Tests for distributed DSC evaluation."""

from __future__ import annotations

import pytest

from wmh_spark.evaluation.metrics import (
    AccuracyGateError,
    EvaluationConfig,
    assert_reproduction_accuracy,
    calculate_dice_result,
)


def test_calculate_dsc_from_spark_masks(spark_session):
    rows = [
        {"postprocessed_mask": 1, "label": 1},
        {"postprocessed_mask": 1, "label": 1},
        {"postprocessed_mask": 1, "label": 0},
        {"postprocessed_mask": 0, "label": 1},
        {"postprocessed_mask": 0, "label": 0},
    ]
    df = spark_session.createDataFrame(rows)

    result = calculate_dice_result(df, EvaluationConfig())

    assert result.predicted_positive == 3
    assert result.reference_positive == 3
    assert result.intersection == 2
    assert result.dsc == pytest.approx(4 / 6)
    assert result.passed is False


def test_accuracy_gate_requires_dsc_greater_than_threshold(spark_session):
    passing_df = spark_session.createDataFrame(
        [{"postprocessed_mask": 1, "label": 1} for _ in range(10)]
        + [{"postprocessed_mask": 0, "label": 0}]
    )
    failing_df = spark_session.createDataFrame(
        [
            {"postprocessed_mask": 1, "label": 1},
            {"postprocessed_mask": 1, "label": 0},
            {"postprocessed_mask": 0, "label": 1},
        ]
    )

    passing = assert_reproduction_accuracy(passing_df, EvaluationConfig())
    assert passing.dsc == pytest.approx(1.0)
    assert passing.passed is True

    with pytest.raises(AccuracyGateError, match="DSC"):
        assert_reproduction_accuracy(failing_df, EvaluationConfig())
