# Implementation Plan: Automated Evaluation & Benchmarking Suite

**Branch**: `006-automated-evaluation-benchmarking` | **Date**: 2026-04-26 | **Spec**: [spec.md](spec.md)

## Summary

Add `wmh_spark.evaluation.metrics` for distributed DSC calculation and accuracy gating, plus `wmh_spark.evaluation.benchmarking` for JSONL benchmark records. The implementation uses Spark aggregations for voxel overlap counts, stores reproducible benchmark dictionaries, compares elapsed time against MATLAB baselines, and summarizes 1-to-4-worker throughput scaling.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: PySpark 3.5.1, stdlib JSON/time/dataclasses
**Storage**: Spark DataFrames for evaluation; JSON Lines files for benchmark logs
**Testing**: pytest with local SparkSession fixture and synthetic mask/benchmark records
**Target Platform**: Local dev + CHPC cluster benchmark runs
**Constraints**: DSC threshold is strict `> 0.85`; benchmark logging must not depend on MATLAB being installed

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Distributed-First | **COMPLIANT** | DSC counts are computed with Spark DataFrame aggregations. |
| II. Reproducible Pre-Processing | **N/A** | This feature evaluates outputs. |
| III. Memory-Safe Partitioning | **N/A** | Evaluation performs reductions over already partitioned DataFrames. |
| IV. Accuracy-Gated Output | **COMPLIANT** | `assert_reproduction_accuracy()` enforces `DSC > 0.85`. |
| V. Scalability & Benchmarking | **COMPLIANT** | Per-subject timing, throughput, and scaling efficiency are logged. |
| VI. Test-Driven Correctness | **COMPLIANT** | Tests cover DSC, threshold gate, benchmark records, throughput, and scaling summary. |

## Project Structure

```text
wmh-spark/
├── src/wmh_spark/
│   └── evaluation/
│       ├── __init__.py
│       ├── benchmarking.py
│       └── metrics.py
└── tests/
    └── unit/
        └── evaluation/
            ├── __init__.py
            ├── test_benchmarking.py
            └── test_metrics.py
```

## Data Flow

1. Feature 5 outputs a DataFrame with `postprocessed_mask`.
2. Ingestion/Feature 3 label path provides expert `label` values.
3. `calculate_dice_result()` aggregates positive prediction count, positive reference count, and intersection count.
4. `assert_reproduction_accuracy()` enforces `DSC > 0.85`.
5. Benchmark functions append JSONL records for subject elapsed time and throughput.
6. Scaling summary compares a baseline worker record against a target worker record.

## Benchmarking Metrics

- `speedup_vs_matlab = matlab_baseline_seconds / elapsed_seconds`
- `subjects_per_hour = subjects_processed / (elapsed_seconds / 3600)`
- `throughput_speedup = target_subjects_per_hour / baseline_subjects_per_hour`
- `scaling_efficiency = throughput_speedup / (target_worker_nodes / baseline_worker_nodes)`
- `near_linear = scaling_efficiency >= 0.75`

## Validation

- Required evaluation columns must exist.
- Empty masks fail because DSC is undefined.
- Elapsed seconds and subjects processed must be positive.
- Worker node counts must be positive.
