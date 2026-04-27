---
description: "Task list for 3D Post-Processing & Connected Component Labeling"
---

# Tasks: 3D Post-Processing & Connected Component Labeling

**Input**: Design documents from `specs/005-3d-post-processing/`
**Prerequisites**: Feature 4 prediction DataFrame with `predicted_mask`
**Tech stack**: Python 3.11, PySpark 3.5.1, NumPy 1.26.4, SciPy 1.11.4, pytest 8.1.1
**Source root**: `wmh-spark/src/wmh_spark/` | **Tests root**: `wmh-spark/tests/`
**Tests**: Included - write failing tests first, then implement

## Phase 1: Spec Kit Scaffold

- [x] T001 Create `specs/005-3d-post-processing/spec.md`
- [x] T002 Create `specs/005-3d-post-processing/plan.md`
- [x] T003 Create `specs/005-3d-post-processing/tasks.md`

## Phase 2: Tests First

- [x] T004 [P] Create `wmh-spark/tests/unit/postprocessing/__init__.py`
- [x] T005 [P] Write `test_reconstruct_prediction_volume_from_flat_dataframe`
- [x] T006 [P] Write `test_connected_components_uses_26_connectivity`
- [x] T007 [P] Write `test_component_sizes_are_calculated`
- [x] T008 [P] Write `test_postprocess_predictions_filters_small_clusters_and_preserves_rows`
- [x] T009 [P] Write `test_postprocess_predictions_requires_prediction_column`

## Phase 3: Implementation

- [x] T010 Implement `PostProcessingConfig`
- [x] T011 Implement `infer_volume_shape()`
- [x] T012 Implement `reconstruct_prediction_volume()`
- [x] T013 Implement `label_connected_components()`
- [x] T014 Implement `component_size_map()`
- [x] T015 Implement `filter_components_by_size()`
- [x] T016 Implement `postprocess_predictions()`
- [x] T017 Export post-processing API from `wmh_spark.postprocessing`

## Phase 4: Verification

- [x] T018 Run focused post-processing tests and confirm they pass
- [x] T019 Run Feature 4 tests to confirm `predicted_mask` compatibility
- [x] T020 Run full local test suite
- [x] T021 Update task checkboxes after implementation

## Notes

- Use 26-connectivity: voxels touching by faces, edges, or corners are grouped together.
- `cluster_size` records the original component size even when a component is removed by thresholding.
- `postprocessed_mask` is the filtered binary output for Feature 6 evaluation and 3D reconstruction.
