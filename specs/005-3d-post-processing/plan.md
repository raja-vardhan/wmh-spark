# Implementation Plan: 3D Post-Processing & Connected Component Labeling

**Branch**: `005-3d-post-processing` | **Date**: 2026-04-26 | **Spec**: [spec.md](spec.md)

## Summary

Add `wmh_spark.postprocessing.connected_components`, a per-subject volumetric post-processing module. It reconstructs Feature 4 `predicted_mask` rows into a `(z, y, x)` NumPy volume, applies 26-connected component labeling with SciPy, calculates voxel counts per cluster, removes clusters below a configurable minimum size, and joins `component_id`, `cluster_size`, and `postprocessed_mask` back onto the original Spark DataFrame.

## Technical Context

**Language/Version**: Python 3.11
**Primary Dependencies**: PySpark 3.5.1, NumPy 1.26.4, SciPy 1.11.4
**Storage**: Spark DataFrame input/output; local NumPy volume for per-subject connected components
**Testing**: pytest with local SparkSession fixture and synthetic prediction tables
**Target Platform**: Local dev and CHPC worker/driver per-subject post-processing stage
**Constraints**: 26-connectivity required; small clusters must be converted to `0`; output must remain tabular for evaluation/reconstruction

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Distributed-First | **COMPLIANT WITH JUSTIFICATION** | The pipeline remains Spark-tabular at boundaries. Connected component labeling is a spatial graph traversal over one subject volume, so a local per-subject NumPy/SciPy step is appropriate after distributed classification. |
| II. Reproducible Pre-Processing | **N/A** | No stochastic preprocessing is introduced. |
| III. Memory-Safe Partitioning | **COMPLIANT** | Post-processing is per subject and only stores compact binary/component volumes. Output joins preserve original partitioning where possible. |
| IV. Accuracy-Gated Output | **PARTIAL** | Removes noise before evaluation; DSC gating remains in Feature 6. |
| V. Scalability & Benchmarking | **PARTIAL** | Work is subject-local and can be parallelized at the subject level in batch orchestration. |
| VI. Test-Driven Correctness | **COMPLIANT** | Tests cover reconstruction, 26-connectivity, size calculation, threshold filtering, and DataFrame row preservation. |

## Project Structure

```text
wmh-spark/
├── src/wmh_spark/
│   └── postprocessing/
│       ├── __init__.py
│       └── connected_components.py
└── tests/
    └── unit/
        └── postprocessing/
            ├── __init__.py
            └── test_connected_components.py
```

## Data Flow

1. Validate `x`, `y`, `z`, and prediction columns.
2. Infer `(z, y, x)` shape from max coordinates or use configured shape.
3. Collect coordinate predictions for one subject and reconstruct a binary volume.
4. Label 26-connected components using `scipy.ndimage.label`.
5. Compute cluster sizes with `numpy.bincount`.
6. Build a coordinate-level result table containing original component IDs, cluster sizes, and thresholded mask values.
7. Join the result table back onto the original Spark DataFrame by `x`, `y`, `z`.

## Validation

- Coordinate and prediction columns must exist.
- Coordinates must be non-negative integers.
- Configured volume shape must contain all coordinates.
- `min_cluster_size` must be positive.
- Input and output row counts should match for one row per voxel coordinate.
