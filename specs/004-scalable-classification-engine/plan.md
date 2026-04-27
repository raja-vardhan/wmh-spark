# Implementation Plan: Scalable Classification Engine

**Branch**: `004-scalable-classification-engine` | **Date**: 2026-04-26 | **Spec**: [spec.md](spec.md)

## Summary

Implement `wmh_spark.models.random_forest`, a Spark MLlib classification module that replaces legacy k-NN behavior with a distributed `RandomForestClassifier`. The module validates Feature 3 input columns, prepares training data with a four-partition default for the CHPC worker layout, trains a Spark ML pipeline, and appends an integer `predicted_mask` column during inference.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: PySpark 3.5.1 Spark MLlib
**Storage**: DataFrames in Spark; model persistence can use Spark ML `PipelineModel.write()` downstream
**Testing**: pytest with the local SparkSession fixture and synthetic Feature 3 DataFrames
**Target Platform**: Local Spark for smoke tests, 4-worker CHPC Spark cluster for project runs
**Constraints**: No scikit-learn or k-NN implementation; labels must be binary; predictions must remain tabular for Feature 5 reconstruction

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Distributed-First | **COMPLIANT** | Uses Spark MLlib `RandomForestClassifier` and DataFrame transformations. |
| II. Reproducible Pre-Processing | **N/A** | This stage consumes preprocessed features; no new preprocessing. |
| III. Memory-Safe Partitioning | **COMPLIANT** | Training DataFrame is explicitly repartitioned to the configured plan and Spark shuffle partitions are aligned. |
| IV. Accuracy-Gated Output | **PARTIAL** | This feature produces raw masks; DSC gates are handled in evaluation/post-processing features. |
| V. Scalability & Benchmarking | **PARTIAL** | The four-worker partition default enables parallel training; detailed throughput benchmarking remains in Feature 6. |
| VI. Test-Driven Correctness | **COMPLIANT** | Tests are added before implementation for training, validation, partitioning, and prediction output. |

## Project Structure

```text
wmh-spark/
├── src/wmh_spark/
│   └── models/
│       ├── __init__.py
│       └── random_forest.py
└── tests/
    └── unit/
        └── models/
            ├── __init__.py
            └── test_random_forest.py
```

## Data Flow

1. Receive a Feature 3 DataFrame with coordinates, raw intensities, derived ratio, spatial prior, and optional label.
2. Validate all configured feature columns and label column exist.
3. Repartition training data to `training_partitions` and set `spark.sql.shuffle.partitions` to match.
4. Build a Spark ML `Pipeline` with `VectorAssembler` and `RandomForestClassifier`.
5. Fit the pipeline over the distributed DataFrame.
6. Transform feature DataFrames with the fitted model and cast Spark's numeric prediction to integer `predicted_mask`.

## Parallelism Strategy

Default `training_partitions=4` matches the four-worker CHPC target. Local tests may still run on `local[2]`, but the DataFrame partition count and shuffle configuration are set to four so the same code path is exercised.

## Validation

- Required Feature 3 columns: `t1`, `flair`, `t1_flair_ratio`, `spatial_prior`
- Required training label column: `label`
- Training labels must be binary (`0` or `1`)
- Prediction output must preserve row count and contain integer `0`/`1` values
