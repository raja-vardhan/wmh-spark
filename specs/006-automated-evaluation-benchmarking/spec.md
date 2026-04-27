# Feature Specification: Automated Evaluation & Benchmarking Suite

**Feature Branch**: `006-automated-evaluation-benchmarking`
**Created**: 2026-04-26
**Status**: Draft
**Input**: User description: "Validate the distributed pipeline's accuracy, scalability, and computational efficiency against the legacy MATLAB baseline."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Calculate Dice Similarity Coefficient (Priority: P1)

An evaluator receives a Spark DataFrame containing the post-processed prediction mask and the expert "Silver Standard" label. The evaluator calculates Dice Similarity Coefficient (DSC) programmatically.

**Why this priority**: DSC is the core reproduction accuracy metric for WMH segmentation.

**Independent Test**: Create a small Spark DataFrame with known prediction/reference overlap and assert the computed DSC equals the hand-calculated value.

**Acceptance Scenarios**:

1. **Given** prediction and reference mask columns, **When** evaluation runs, **Then** it calculates `DSC = 2 * intersection / (prediction positives + reference positives)`.
2. **Given** both masks are empty, **When** DSC is requested, **Then** the evaluator fails fast because DSC is undefined for empty masks.

---

### User Story 2 - Enforce Reproduction Accuracy Threshold (Priority: P1)

The test/evaluation suite asserts that the final post-processed mask reproduces the expert mask with `DSC > 0.85`.

**Why this priority**: The project requires objective correctness before benchmarking or reporting performance.

**Independent Test**: Provide a high-overlap synthetic mask and assert it passes; provide a low-overlap mask and assert threshold failure.

**Acceptance Scenarios**:

1. **Given** `DSC > 0.85`, **When** the accuracy gate runs, **Then** it returns a passing result.
2. **Given** `DSC <= 0.85`, **When** the accuracy gate runs, **Then** it raises a descriptive error.

---

### User Story 3 - Benchmark End-to-End Time Per Subject (Priority: P1)

The benchmark suite logs per-subject end-to-end processing time and compares it with a legacy MATLAB baseline duration.

**Why this priority**: The migration must prove reduced latency relative to the serial MATLAB implementation.

**Independent Test**: Log a synthetic subject timing record and assert elapsed time, baseline time, speedup, and reduced-latency status are written to JSON Lines.

**Acceptance Scenarios**:

1. **Given** an end-to-end elapsed time and MATLAB baseline, **When** benchmark logging runs, **Then** a JSONL record stores elapsed seconds, baseline seconds, and speedup.
2. **Given** elapsed time less than baseline time, **When** benchmark logging runs, **Then** the record marks reduced latency as true.

---

### User Story 4 - Track Cluster Throughput Scaling (Priority: P2)

The benchmark suite records subjects processed per hour at different worker counts and computes scaling efficiency from 1 to 4 workers.

**Why this priority**: The CHPC objective is near-linear scalability as the cluster scales from 1 to 4 worker nodes.

**Independent Test**: Log 1-worker and 4-worker synthetic throughput records and assert 4-worker efficiency is near-linear when throughput is at least 75% of ideal linear speedup.

**Acceptance Scenarios**:

1. **Given** subjects processed and elapsed seconds, **When** throughput logging runs, **Then** the benchmark record includes subjects per hour.
2. **Given** 1-worker and 4-worker throughput records, **When** scaling is summarized, **Then** the suite reports speedup and scaling efficiency.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The evaluator MUST compute DSC between Spark-generated post-processed masks and expert labels.
- **FR-002**: The evaluator MUST enforce `DSC > 0.85` as the reproduction accuracy threshold.
- **FR-003**: The benchmark suite MUST log end-to-end elapsed seconds per subject.
- **FR-004**: The benchmark suite MUST compare per-subject elapsed seconds against a MATLAB baseline and record speedup.
- **FR-005**: The benchmark suite MUST calculate and log cluster throughput in subjects per hour.
- **FR-006**: The benchmark suite MUST summarize 1-to-4-worker scaling speedup and efficiency.
- **FR-007**: Evaluation and benchmark records MUST be structured dictionaries suitable for JSON Lines output.

### Key Entities

- **DiceResult**: DSC, threshold, pass/fail status, positive voxel counts, and intersection count.
- **SubjectBenchmarkRecord**: Subject ID, elapsed seconds, MATLAB baseline seconds, speedup, reduced-latency flag, worker count, and timestamp.
- **ThroughputRecord**: Worker count, subjects processed, elapsed seconds, subjects per hour, and timestamp.
- **ScalingSummary**: Baseline and target worker counts, throughput speedup, ideal speedup, scaling efficiency, and near-linear flag.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Synthetic Spark masks with known overlap produce exact DSC values.
- **SC-002**: Accuracy gate passes only when `DSC > 0.85`.
- **SC-003**: Per-subject benchmark JSONL records include speedup versus MATLAB.
- **SC-004**: Throughput records include subjects per hour.
- **SC-005**: Scaling summaries report near-linear scalability for 4-worker throughput at least 75% of ideal.

## Assumptions

- The evaluation DataFrame contains one row per voxel with expert labels in `label` and post-processed predictions in `postprocessed_mask`.
- MATLAB baseline durations are supplied as configuration or run metadata; the suite records and compares them but does not run MATLAB.
- Throughput scaling can be measured from batch runs at 1 and 4 workers and summarized from the resulting records.
