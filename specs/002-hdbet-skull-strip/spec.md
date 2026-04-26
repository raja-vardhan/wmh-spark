# Feature Specification: HD-BET Skull-Stripping Preprocessing Module

**Feature Branch**: `002-hdbet-skull-strip`
**Created**: 2026-04-26
**Status**: Draft

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Batch Skull-Strip Raw MRI Scan Pairs (Priority: P1)

A pipeline engineer provides a batch of raw T1 and FLAIR NIfTI scan pairs. The module runs local HD-BET on each scan, producing brain-only volumes and binary brain masks. Successfully stripped scans are passed to downstream PySpark feature extraction; any scan failing the quality threshold is quarantined with a logged reason.

**Why this priority**: This is the core function of the module and a hard prerequisite for WMH classification — no other pipeline stage can proceed without skull-stripped inputs.

**Independent Test**:
- *(Phase 1 / smoke-test)*: Structural assertions pass — brain mask is binary, output shape and affine match raw input, brain mask foreground fraction is between 5% and 95%.
- *(Phase 2 / Kaggle)*: DSC of each skull-stripped output against its expert reference mask exceeds 0.85.

**Acceptance Scenarios**:

1. **Given** a directory of valid raw T1 and FLAIR NIfTI scan pairs, **When** the module runs, **Then** each scan produces a skull-stripped NIfTI volume and a corresponding binary brain mask with all non-brain voxels set to zero.
2. **Given** a batch of scans, **When** processing completes, **Then** the total voxel count of each output is at least 60% lower than its corresponding raw input volume.
3. **Given** a scan pair where one scan fails the DSC quality gate (DSC ≤ 0.85), **When** processing completes, **Then** both scans in the pair are quarantined and excluded from downstream processing, with the failure reason logged.

---

### User Story 2 — Quality-Gate Validation Before Downstream Handoff (Priority: P2)

After skull-stripping, the module evaluates each output against the project's DSC quality threshold. Only scans meeting the threshold proceed to PySpark feature extraction; failing scans are flagged for review without blocking the rest of the batch.

**Why this priority**: Passing confounded or poorly stripped scans to the classifier degrades model performance. The quality gate is a mandatory checkpoint between preprocessing and feature extraction.

**Independent Test**: Inject a deliberately poor skull-strip output (DSC ≈ 0.60) alongside valid ones; confirm the failing scan is excluded and logged while valid scans proceed unaffected.

**Acceptance Scenarios**:

1. **Given** a skull-stripped output with DSC > 0.85 against its reference mask, **When** the quality gate runs, **Then** the scan is marked as accepted and forwarded to the feature extraction stage.
2. **Given** a skull-stripped output with DSC ≤ 0.85, **When** the quality gate runs, **Then** the scan is marked as rejected, excluded from feature extraction, and a structured log entry is written with the scan ID and measured DSC value.
3. **Given** a batch with mixed passing and failing scans, **When** the quality gate runs, **Then** passing scans proceed without delay and failing scans do not block the rest of the batch.

---

### User Story 3 — Single-Scan Reprocessing for Debugging (Priority: P3)

A pipeline maintainer needs to reprocess a single T1/FLAIR pair — either to debug a failure or verify a fix — without rerunning the entire batch.

**Why this priority**: Operational necessity for debugging and remediation, but the batch workflow remains the primary path.

**Independent Test**: Run the module on a single scan pair by specifying its path; confirm the output skull-stripped volumes and mask are produced correctly with no side effects on other scans.

**Acceptance Scenarios**:

1. **Given** a single raw T1/FLAIR scan pair path, **When** the module is invoked in single-scan mode, **Then** only that pair is processed and the output is written to the designated output directory.
2. **Given** a previously failed scan that has been corrected, **When** it is reprocessed individually, **Then** the new output overwrites the previous output and the log reflects the updated result.

---

### Edge Cases

- What happens when a scan has severe motion artifacts that cause HD-BET to produce an incomplete brain mask?
- How does the module handle NIfTI files with non-standard or missing orientation headers?
- What if the local HD-BET executable is unavailable on the execution node?
- What happens when one scan in a T1/FLAIR pair is missing or corrupted?
- How are duplicate scan IDs handled if the same scan appears more than once in the input batch?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The module MUST accept a batch of raw T1 and FLAIR NIfTI scan pairs (`.nii` or `.nii.gz`) as input, specified via a directory path or a manifest file.
- **FR-002**: The module MUST apply HD-BET skull-stripping independently to every T1 and FLAIR scan in the input batch.
- **FR-003**: HD-BET MUST be executed via the local pinned dependency stack; no Docker daemon or container image shall be required.
- **FR-004**: The module MUST produce two outputs per scan: a skull-stripped NIfTI volume (non-brain voxels zeroed) and a binary brain mask NIfTI volume.
- **FR-005**: The module MUST validate each skull-stripped output against a DSC quality threshold of 0.85; outputs below the threshold MUST be quarantined and excluded from downstream handoff.
- **FR-006**: The module MUST write a structured processing log per run recording one entry per scan pair: scan pair ID, processing status (accepted/rejected), measured DSC value (or null in smoke-test mode), and failure reason where applicable.
- **FR-007**: The module MUST support single-scan reprocessing mode, allowing a specific T1/FLAIR pair to be reprocessed individually without affecting other outputs.
- **FR-008**: The module MUST be deterministic — identical input scans MUST produce bit-identical outputs across runs on the same pinned HD-BET dependency versions.
- **FR-009**: The module MUST handle individual scan failures gracefully: a failure in one scan pair MUST NOT halt processing of the remaining batch.

### Key Entities

- **Raw MRI Scan**: Input NIfTI volume (T1 or FLAIR) containing full head anatomy including skull, eyes, skin, and background signal.
- **Scan Pair**: A coupled T1 and FLAIR scan belonging to the same subject and acquisition session; treated as an atomic unit for quality-gating purposes.
- **Brain Mask**: Binary NIfTI volume produced by HD-BET; value of 1 indicates brain tissue, 0 indicates non-brain.
- **Skull-Stripped Volume**: NIfTI volume derived by applying the brain mask to the raw scan; all non-brain voxels are set to zero.
- **Processing Log**: Structured record (one entry per scan) capturing scan ID, DSC score, accept/reject decision, and any error details.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of raw T1 and FLAIR scans submitted to the module are processed by HD-BET; no scans are silently skipped.
- **SC-002**: Skull-stripped outputs achieve DSC > 0.85 compared to reference brain masks on the project's validation scan set.
- **SC-003**: Output voxel count is reduced by at least 60% relative to the raw input for every accepted scan, confirming removal of skull and background.
- **SC-004**: Processing is deterministic — re-running the module on the same input and pinned HD-BET dependency versions produces bit-identical outputs.
- **SC-005**: Any scan failing the DSC quality gate is excluded from downstream feature extraction with zero false passes.
- **SC-006**: A complete processing log is available at the end of every batch run, with one entry per scan pair including accept/reject status and DSC value.

## Assumptions

- Input NIfTI files are already in standard orientation and do not require reorientation prior to skull-stripping.
- T1 and FLAIR scans are skull-stripped independently (no co-registration required before stripping).
- The HD-BET CLI is installed and available on the execution nodes before the module runs.
- CPU execution is supported for local smoke tests; GPU execution can be selected with `--device cuda` when the local PyTorch stack supports CUDA.
- Reference brain masks for DSC validation are available for the project's validation scan set (Phase 2 / Kaggle). In smoke-test mode (Phase 1, local `datasets/`), no reference masks are available and the DSC gate is skipped entirely; structural assertions replace it (binary mask, shape/affine match, foreground fraction 5–95%).
- Output skull-stripped volumes are written to a configurable output directory that is accessible to the downstream PySpark feature extraction stage.
- The DSC threshold of 0.85 is fixed per the project's existing quality standard and is not configurable per-run.
