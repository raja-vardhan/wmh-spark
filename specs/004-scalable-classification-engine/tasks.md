---
description: "Task list for Scalable Classification Engine"
---

# Tasks: Scalable Classification Engine

**Input**: Design documents from `specs/004-scalable-classification-engine/`
**Prerequisites**: Feature 3 distributed voxel feature DataFrame
**Tech stack**: Python 3.11, PySpark 3.5.1 Spark MLlib, pytest 8.1.1
**Source root**: `wmh-spark/src/wmh_spark/` | **Tests root**: `wmh-spark/tests/`
**Tests**: Included - write failing tests first, then implement

## Phase 1: Spec Kit Scaffold

- [x] T001 Create `specs/004-scalable-classification-engine/spec.md`
- [x] T002 Create `specs/004-scalable-classification-engine/plan.md`
- [x] T003 Create `specs/004-scalable-classification-engine/tasks.md`

## Phase 2: Tests First

- [x] T004 [P] Create `wmh-spark/tests/unit/models/__init__.py`
- [x] T005 [P] Write `test_prepare_training_dataframe_uses_feature3_columns_and_partitions`
- [x] T006 [P] Write `test_train_random_forest_uses_spark_mllib_model`
- [x] T007 [P] Write `test_predict_voxel_mask_outputs_binary_column_and_preserves_rows`
- [x] T008 [P] Write `test_training_requires_label_column`
- [x] T009 [P] Write `test_training_rejects_non_binary_labels`

## Phase 3: Implementation

- [x] T010 Implement `ClassificationConfig`
- [x] T011 Implement `validate_feature_columns()`
- [x] T012 Implement `prepare_training_dataframe()`
- [x] T013 Implement `train_random_forest_model()`
- [x] T014 Implement `predict_voxel_mask()`
- [x] T015 Export the classifier API from `wmh_spark.models`

## Phase 4: Verification

- [x] T016 Run focused Random Forest tests and confirm they pass
- [x] T017 Run Feature 3 tests to confirm classifier integration inputs remain stable
- [x] T018 Run full local test suite
- [x] T019 Update task checkboxes after implementation

## Notes

- The model must be Spark MLlib Random Forest, not k-NN and not scikit-learn.
- Default training parallelism is four partitions for the four-worker CHPC target.
- The raw classifier output column for Feature 5 is `predicted_mask`.
