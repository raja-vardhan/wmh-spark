# Implementation Plan: Distributed Feature Extraction Engine

**Branch**: `003-distributed-feature-extraction` | **Date**: 2026-04-26 | **Spec**: [spec.md](spec.md)

## Summary

Add a `wmh_spark.feature_extraction` module that builds on the existing `io_utils.build_voxel_dataframe()` ingestion primitive. The module resolves a memory-safe feature partition count, builds the flattened voxel DataFrame, adds `t1_flair_ratio` with Spark SQL expressions, loads a standard spatial-prior NIfTI as a coordinate table, and joins that prior with a broadcast join to avoid shuffling the full subject feature matrix.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: PySpark 3.5.1, NumPy 1.26.4, NiBabel 5.2.1, pandas 2.2.1
**Storage**: NIfTI files on local filesystem / CHPC mounted volume
**Testing**: pytest with local SparkSession fixture
**Target Platform**: Linux local dev + University of Utah CHPC Spark workers
**Performance Goals**: Avoid Python voxel loops after initial NIfTI flattening; keep estimated shuffle partitions under a conservative target and the 16 GB worker limit
**Constraints**: T1, FLAIR, and prior template must share shape and affine; Spark partitioning rationale must be documented at repartition call sites

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Distributed-First | **COMPLIANT** | Ratio and coordinate prior mapping are DataFrame transformations. NIfTI loading remains the existing accepted local boundary. |
| II. Reproducible Pre-Processing | **COMPLIANT** | No new stochastic preprocessing is introduced. Inputs are validated by shape and affine. |
| III. Memory-Safe Partitioning | **COMPLIANT** | Feature partition count is derived from estimated bytes per row, a 256 MiB shuffle target, and the 16 GB hard worker ceiling. |
| IV. Accuracy-Gated Output | **N/A** | This stage creates features; model accuracy gates apply downstream. |
| V. Scalability & Benchmarking | **PARTIAL** | The module logs partition planning. Full cluster throughput metrics remain in the benchmark suite feature. |
| VI. Test-Driven Correctness | **COMPLIANT** | Tests are added before implementation and cover schema, ratio, spatial prior mapping, validation, and partition planning. |

## Project Structure

```text
wmh-spark/
├── src/wmh_spark/
│   └── feature_extraction.py             # FeatureDataFrame builder + partition planning
└── tests/
    └── unit/
        └── test_feature_extraction.py    # Unit tests using synthetic volumes + Spark
```

## Data Flow

1. Validate the spatial-prior template against the subject T1 grid.
2. Resolve feature partition count from voxel count or explicit override.
3. Build the base voxel DataFrame with `subject_id`, `x`, `y`, `z`, `t1`, `flair`, and optional `label`.
4. Add `t1_flair_ratio` using `pyspark.sql.functions.when` and column arithmetic.
5. Build a spatial-prior coordinate DataFrame from the template NIfTI.
6. Broadcast join the spatial-prior DataFrame on `x`, `y`, `z` so the large subject DataFrame is not shuffled.

## Partition Strategy

The feature engine uses `derive_feature_partition_count()` with these defaults:

- `bytes_per_row=48`: conservative estimate for coordinates, raw intensities, ratio, prior, optional label, and row overhead.
- `target_partition_bytes=256 MiB`: preferred shuffle partition target.
- `max_partition_bytes=16 GiB`: hard worker-node memory ceiling.
- `min_partitions=4`: keeps local and small-volume tests from collapsing all work into one partition.

Explicit overrides are respected for local smoke tests and cluster tuning. The builder also sets `spark.sql.shuffle.partitions` to the resolved count so any fallback shuffle uses the same memory plan.

## Validation

- Spatial prior shape must exactly match subject shape.
- Spatial prior affine must match subject affine within `atol=1e-4`.
- FLAIR values at or below `ratio_epsilon` in absolute value produce ratio `0.0`.
- Broadcast prior join should keep the streamed subject DataFrame partitioning intact.
