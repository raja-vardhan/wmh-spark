"""Tests for benchmark and scalability logging."""

from __future__ import annotations

import json

import pytest

from wmh_spark.evaluation.benchmarking import (
    log_cluster_throughput,
    log_subject_benchmark,
    summarize_scaling,
)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_subject_benchmark_logs_speedup_against_matlab(tmp_path):
    out = tmp_path / "subject_bench.jsonl"

    record = log_subject_benchmark(
        subject_id="subj-001",
        elapsed_seconds=25.0,
        matlab_baseline_seconds=100.0,
        worker_nodes=4,
        output_path=out,
    )

    assert record["speedup_vs_matlab"] == pytest.approx(4.0)
    assert record["reduced_latency"] is True
    assert record["worker_nodes"] == 4
    assert _read_jsonl(out)[0]["subject_id"] == "subj-001"


def test_throughput_logging_calculates_subjects_per_hour(tmp_path):
    out = tmp_path / "throughput.jsonl"

    record = log_cluster_throughput(
        worker_nodes=4,
        subjects_processed=8,
        elapsed_seconds=1800.0,
        output_path=out,
    )

    assert record["subjects_per_hour"] == pytest.approx(16.0)
    assert _read_jsonl(out)[0]["subjects_processed"] == 8


def test_scaling_summary_confirms_near_linear_four_worker_scaling():
    one_worker = {
        "worker_nodes": 1,
        "subjects_per_hour": 10.0,
    }
    four_workers = {
        "worker_nodes": 4,
        "subjects_per_hour": 36.0,
    }

    summary = summarize_scaling(
        baseline_record=one_worker,
        target_record=four_workers,
        near_linear_threshold=0.75,
    )

    assert summary["throughput_speedup"] == pytest.approx(3.6)
    assert summary["ideal_speedup"] == pytest.approx(4.0)
    assert summary["scaling_efficiency"] == pytest.approx(0.9)
    assert summary["near_linear"] is True
