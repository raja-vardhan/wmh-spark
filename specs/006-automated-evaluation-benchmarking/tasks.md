---
description: "Task list for Automated Evaluation & Benchmarking Suite"
---

# Tasks: Automated Evaluation & Benchmarking Suite

**Input**: Design documents from `specs/006-automated-evaluation-benchmarking/`
**Prerequisites**: Feature 5 post-processed DataFrame with `postprocessed_mask` and expert `label`
**Tech stack**: Python 3.11, PySpark 3.5.1, pytest 8.1.1
**Source root**: `wmh-spark/src/wmh_spark/` | **Tests root**: `wmh-spark/tests/`
**Tests**: Included - write failing tests first, then implement

## Phase 1: Spec Kit Scaffold

- [x] T001 Create `specs/006-automated-evaluation-benchmarking/spec.md`
- [x] T002 Create `specs/006-automated-evaluation-benchmarking/plan.md`
- [x] T003 Create `specs/006-automated-evaluation-benchmarking/tasks.md`

## Phase 2: Tests First

- [x] T004 [P] Create `wmh-spark/tests/unit/evaluation/__init__.py`
- [x] T005 [P] Write `test_calculate_dsc_from_spark_masks`
- [x] T006 [P] Write `test_accuracy_gate_requires_dsc_greater_than_threshold`
- [x] T007 [P] Write `test_subject_benchmark_logs_speedup_against_matlab`
- [x] T008 [P] Write `test_throughput_logging_calculates_subjects_per_hour`
- [x] T009 [P] Write `test_scaling_summary_confirms_near_linear_four_worker_scaling`

## Phase 3: Implementation

- [x] T010 Implement `EvaluationConfig` and `DiceResult`
- [x] T011 Implement `calculate_dice_result()`
- [x] T012 Implement `assert_reproduction_accuracy()`
- [x] T013 Implement `SubjectBenchmarkRecord` and `log_subject_benchmark()`
- [x] T014 Implement `ThroughputRecord` and `log_cluster_throughput()`
- [x] T015 Implement `summarize_scaling()`
- [x] T016 Export evaluation APIs from `wmh_spark.evaluation`

## Phase 4: Verification

- [x] T017 Run focused evaluation tests and confirm they pass
- [x] T018 Run Feature 5 tests to confirm `postprocessed_mask` compatibility
- [x] T019 Run full local test suite
- [x] T020 Update task checkboxes after implementation

## Notes

- DSC threshold is strict: `DSC > 0.85` passes, `DSC <= 0.85` fails.
- Benchmark records are JSON Lines for easy append-only cluster logging.
- MATLAB baseline values are provided by metadata; the suite records comparison metrics but does not invoke MATLAB.
