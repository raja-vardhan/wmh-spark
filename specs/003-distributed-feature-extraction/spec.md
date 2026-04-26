# Feature Specification: Distributed Feature Extraction Engine

**Feature Branch**: `003-distributed-feature-extraction`
**Created**: 2026-04-26
**Status**: Draft
**Input**: User description: "Translate sequential MATLAB feature generation into a vectorized, distributed PySpark architecture."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build a Distributed Voxel Feature Table (Priority: P1)

A pipeline engineer provides paired skull-stripped T1 and FLAIR NIfTI scans for one subject. The engine flattens the 3D arrays into a partitioned PySpark DataFrame where each row is one voxel and includes coordinates plus raw intensity values.

**Why this priority**: This is the required handoff format for Spark MLlib training and prediction.

**Independent Test**: Use the synthetic subject fixture and assert the DataFrame has one row per voxel with `subject_id`, `x`, `y`, `z`, `t1`, and `flair` columns.

**Acceptance Scenarios**:

1. **Given** co-registered T1 and FLAIR volumes, **When** the feature engine runs, **Then** it emits a Spark DataFrame with one row per `(x, y, z)` voxel.
2. **Given** the emitted DataFrame, **When** downstream code reads it, **Then** raw T1 and FLAIR intensity columns are available without requiring another NIfTI read.

---

### User Story 2 - Calculate T1/FLAIR Ratio with Spark Expressions (Priority: P1)

For every voxel, the engine computes a `t1_flair_ratio` feature using distributed Spark SQL column expressions rather than Python voxel loops or Python UDFs.

**Why this priority**: The MATLAB implementation loops voxel-by-voxel; replacing that loop with vectorized Spark operations is the central migration goal.

**Independent Test**: Build a feature DataFrame and assert sampled ratio values equal `t1 / flair`; inspect the physical plan to ensure no Python UDF evaluation stage is present.

**Acceptance Scenarios**:

1. **Given** a voxel with non-zero FLAIR intensity, **When** features are computed, **Then** `t1_flair_ratio = t1 / flair`.
2. **Given** a voxel with zero or near-zero FLAIR intensity, **When** features are computed, **Then** `t1_flair_ratio` is set to `0.0` to avoid `NaN` and `Infinity` values in MLlib inputs.

---

### User Story 3 - Append Spatial Prior from Standard Template (Priority: P2)

The engine maps each voxel coordinate to a standard anatomical probability template and appends the template value as `spatial_prior`.

**Why this priority**: The spatial prior is a core UBO Detector feature and improves lesion classification by injecting anatomical likelihood.

**Independent Test**: Save a synthetic spatial-prior NIfTI on the same grid as the subject, run the engine, and assert known `(x, y, z)` coordinates receive the expected prior values.

**Acceptance Scenarios**:

1. **Given** a standard prior template on the same voxel grid as the subject volumes, **When** the feature engine runs, **Then** every voxel row includes the matching `spatial_prior` value.
2. **Given** a prior template with mismatched shape or affine transform, **When** the feature engine validates inputs, **Then** it raises a descriptive `ValueError` and produces no partial feature DataFrame.

---

### User Story 4 - Memory-Safe Shuffle Partition Planning (Priority: P1)

The engine derives a conservative partition count for feature-generation and coordinate-join operations so no shuffle partition approaches the 16 GB worker memory limit.

**Why this priority**: The CHPC target has only 16 GB RAM per worker node, so the Spark plan must avoid oversized shuffle partitions.

**Independent Test**: Assert the derived partition count keeps estimated bytes per partition below the target shuffle size and the 16 GB hard ceiling; assert explicit overrides are respected.

**Acceptance Scenarios**:

1. **Given** no explicit partition count, **When** the engine receives a voxel count, **Then** it derives a partition count using a conservative shuffle target below the 16 GB worker ceiling.
2. **Given** an explicit partition count, **When** the engine runs, **Then** the output DataFrame uses that count and Spark SQL shuffle partitions are set to the same value for downstream operations.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The engine MUST flatten paired 3D T1 and FLAIR arrays into a 2D PySpark DataFrame with one row per `(x, y, z)` voxel.
- **FR-002**: The engine MUST include raw `t1` and `flair` intensity columns in the output DataFrame.
- **FR-003**: The engine MUST compute `t1_flair_ratio` using Spark column expressions, not Python voxel loops or Python UDFs.
- **FR-004**: The engine MUST append `spatial_prior` by mapping voxel coordinates to a standard template.
- **FR-005**: The engine MUST reject spatial-prior templates whose shape or affine does not match the subject grid.
- **FR-006**: The engine MUST derive a memory-safe partition count when no explicit count is supplied.
- **FR-007**: The engine MUST set Spark SQL shuffle partitions to the resolved feature partition count during feature generation.
- **FR-008**: The engine MUST support optional lesion mask labels from the existing ingestion layer without changing label semantics.

### Key Entities

- **Voxel Feature Record**: One Spark row with subject ID, voxel coordinates, raw intensities, derived ratio, spatial prior, and optional label.
- **Spatial Prior Template**: A NIfTI probability volume in the same standard grid as subject features; values represent anatomical lesion prior probability.
- **Feature Partition Plan**: The resolved partition count and byte estimates used to keep shuffle work below the worker memory budget.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Synthetic test volumes produce exactly `shape[0] * shape[1] * shape[2]` feature rows.
- **SC-002**: Sampled `t1_flair_ratio` values are numerically correct within `1e-6` tolerance for non-zero FLAIR voxels.
- **SC-003**: Spatial-prior values joined by coordinate match the synthetic template exactly for sampled voxels.
- **SC-004**: The Spark plan for ratio generation contains no Python UDF evaluation stage.
- **SC-005**: Auto-derived partition plans keep estimated partition size below the configured shuffle target and far below 16 GB per worker.

## Assumptions

- T1, FLAIR, and spatial-prior volumes are already registered to the same voxel grid before feature extraction.
- The existing ingestion layer remains responsible for loading, validating, and flattening T1/FLAIR volumes.
- Spatial prior mapping in this feature uses direct voxel-coordinate lookup on a matching template grid; non-linear registration is out of scope.
- Local smoke tests use small synthetic NIfTI volumes; CHPC runs use the same API with larger partition counts.
