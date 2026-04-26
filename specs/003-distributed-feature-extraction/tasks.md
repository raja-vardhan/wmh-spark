---
description: "Task list for Distributed Feature Extraction Engine"
---

# Tasks: Distributed Feature Extraction Engine

**Input**: Design documents from `specs/003-distributed-feature-extraction/`
**Prerequisites**: `io_utils.build_voxel_dataframe()` from Feature 1
**Tech stack**: Python 3.11, PySpark 3.5.1, NiBabel 5.2.1, NumPy 1.26.4, pytest 8.1.1
**Source root**: `wmh-spark/src/wmh_spark/` | **Tests root**: `wmh-spark/tests/`
**Tests**: Included - test first, then implement

## Phase 1: Spec Kit Scaffold

- [x] T001 Create `specs/003-distributed-feature-extraction/spec.md`
- [x] T002 Create `specs/003-distributed-feature-extraction/plan.md`
- [x] T003 Create `specs/003-distributed-feature-extraction/tasks.md`

## Phase 2: Tests First

- [x] T004 [P] Write `test_build_feature_dataframe_schema_ratio_and_spatial_prior` in `wmh-spark/tests/unit/test_feature_extraction.py`
- [x] T005 [P] Write `test_feature_ratio_uses_spark_expression_not_python_udf`
- [x] T006 [P] Write `test_spatial_prior_template_shape_mismatch_fails_fast`
- [x] T007 [P] Write `test_derive_feature_partition_count_keeps_partitions_under_shuffle_target`
- [x] T008 [P] Write `test_build_feature_dataframe_applies_partition_plan`

## Phase 3: Implementation

- [x] T009 Implement `FeaturePartitionPlan` and `derive_feature_partition_count()`
- [x] T010 Implement `validate_spatial_prior_template()`
- [x] T011 Implement `build_spatial_prior_dataframe()`
- [x] T012 Implement `add_t1_flair_ratio()`
- [x] T013 Implement `build_feature_dataframe()`
- [x] T014 Export the feature-extraction API from `wmh_spark.__init__`

## Phase 4: Verification

- [x] T015 Run focused feature-extraction tests and confirm they pass
- [x] T016 Run existing ingestion tests to confirm no regression
- [x] T017 Update task checkboxes after implementation

## Dependencies & Execution Order

- T001-T003 first to establish Spec Kit traceability.
- T004-T008 before implementation to satisfy TDD.
- T009-T013 can be implemented in one module after tests fail.
- T015-T016 after implementation.

## Notes

- Constitution Principle III: every `.repartition()` call must document the partition-count rationale.
- The spatial-prior coordinate DataFrame is broadcast during the join to avoid shuffling the larger subject feature matrix.
- Non-linear template registration is out of scope for this feature; all inputs must already be on the same standard grid.
