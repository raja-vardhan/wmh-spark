# Feature Specification: 3D Post-Processing & Connected Component Labeling

**Feature Branch**: `005-3d-post-processing`
**Created**: 2026-04-26
**Status**: Draft
**Input**: User description: "Filter the raw classification output using volumetric spatial logic to eliminate scanner noise and false positives."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Reconstruct Voxel Predictions into a 3D Volume (Priority: P1)

A downstream pipeline stage receives a flattened Spark DataFrame from Feature 4 with voxel coordinates and a raw binary `predicted_mask` column. The post-processing module reconstructs those rows into a localized 3D prediction matrix.

**Why this priority**: Connected component labeling requires 3D spatial neighborhood relationships; the flattened table must be restored to a volume.

**Independent Test**: Build a small Spark DataFrame with known `(x, y, z)` coordinates and assert the reconstructed NumPy volume has matching `volume[z, y, x]` values.

**Acceptance Scenarios**:

1. **Given** a flattened prediction DataFrame, **When** reconstruction runs, **Then** the output volume shape matches the inferred or configured `(z, y, x)` grid.
2. **Given** a voxel row with `predicted_mask=1`, **When** reconstruction runs, **Then** `volume[z, y, x]` is `1`.

---

### User Story 2 - Label 26-Connected 3D Lesion Components (Priority: P1)

The module groups adjacent positive voxels into connected components where adjacency includes faces, edges, and corners.

**Why this priority**: WMH lesions are spatial clusters, and 26-connectivity matches the PRD requirement for face/edge/corner contact.

**Independent Test**: Create two diagonal voxels touching only by a corner and assert they are labeled as one component; create a separate distant voxel and assert it becomes another component.

**Acceptance Scenarios**:

1. **Given** positive voxels touching by a face, edge, or corner, **When** labeling runs, **Then** they share one component ID.
2. **Given** positive voxels separated in 3D space, **When** labeling runs, **Then** they receive different component IDs.

---

### User Story 3 - Calculate Cluster Size for Each Component (Priority: P1)

The module calculates each connected component's size in voxels and writes the size back to the voxel-level DataFrame.

**Why this priority**: The minimum-volume threshold is defined in voxel count, so each cluster must be measured before filtering.

**Independent Test**: Label a synthetic volume with component sizes 3 and 1, then assert the component-size mapping and output DataFrame `cluster_size` values match.

**Acceptance Scenarios**:

1. **Given** a labeled 3D prediction volume, **When** cluster statistics are calculated, **Then** each component ID has the correct voxel count.
2. **Given** a post-processed output DataFrame, **When** rows are inspected, **Then** positive raw-prediction rows include their original component size.

---

### User Story 4 - Remove Small Components Below Threshold (Priority: P1)

The module applies a minimum cluster-size threshold and converts clusters below the threshold back to healthy/noise (`0`).

**Why this priority**: Tiny isolated predictions are common scanner/model noise and should not enter final WMH quantification.

**Independent Test**: Use a synthetic prediction table containing a size-3 component and a size-1 component with `min_cluster_size=2`; assert only the size-3 component remains in `postprocessed_mask`.

**Acceptance Scenarios**:

1. **Given** a component with size below `min_cluster_size`, **When** filtering runs, **Then** all voxels in that component have `postprocessed_mask=0`.
2. **Given** a component with size at least `min_cluster_size`, **When** filtering runs, **Then** all voxels in that component keep `postprocessed_mask=1`.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST reconstruct flattened Feature 4 prediction rows into a 3D matrix.
- **FR-002**: The system MUST perform 3D connected component labeling with 26-connectivity.
- **FR-003**: The system MUST calculate voxel count for each connected component.
- **FR-004**: The system MUST enforce a configurable minimum cluster-size threshold.
- **FR-005**: The system MUST append `component_id`, `cluster_size`, and `postprocessed_mask` columns to the voxel-level Spark DataFrame.
- **FR-006**: The system MUST preserve the input row count during post-processing.
- **FR-007**: The system MUST fail fast if required coordinate or prediction columns are missing.

### Key Entities

- **PostProcessingConfig**: Configuration for prediction column name, output column names, minimum cluster size, and optional target volume shape.
- **Connected Component**: One 26-connected group of predicted lesion voxels in 3D space.
- **Cluster Size**: Total voxel count belonging to one connected component.
- **Post-Processed Mask**: Filtered binary mask after small components are removed.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A known flattened prediction table reconstructs to the expected 3D matrix.
- **SC-002**: Corner-touching voxels are labeled as one component under 26-connectivity.
- **SC-003**: Component size calculations match exact voxel counts.
- **SC-004**: Components below the configured threshold are removed.
- **SC-005**: The post-processed DataFrame preserves the original row count and appends the expected output columns.

## Assumptions

- Post-processing runs per subject after classification.
- Volumetric connected component labeling is local per subject because 3D neighborhood traversal is not naturally represented as a Spark SQL operation.
- DataFrames contain integer voxel coordinates named `x`, `y`, and `z`.
