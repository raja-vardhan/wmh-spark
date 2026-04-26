# Feature Specification: Distributed MRI Data Ingestion & I/O Layer

**Feature Branch**: `001-mri-data-ingestion`
**Created**: 2026-04-25
**Status**: Draft
**Input**: User description: "Securely load and parse heterogeneous 3D MRI sequences into the distributed Spark ecosystem."

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Load Paired MRI Sequences for a Single Subject (Priority: P1)

A pipeline operator provides a subject directory containing a T1-weighted scan and a FLAIR scan
in NIfTI format. The system loads both files, validates they are correctly paired, and makes them
available in the distributed pipeline as a partitioned DataFrame ready for feature extraction.

**Why this priority**: This is the foundational input to the entire pipeline. Nothing downstream
can function without correct ingestion of T1 and FLAIR sequences. This story is the MVP of the
I/O layer.

**Independent Test**: Can be fully tested by pointing the ingestion module at the local
`datasets/` directory (T1_RMS.nii.gz + FLAIR.nii.gz) and asserting that the resulting DataFrame
has the correct schema and non-zero row count.

**Acceptance Scenarios**:

1. **Given** a subject directory with a valid T1 NIfTI file and a valid FLAIR NIfTI file,
   **When** the ingestion module is run against that directory,
   **Then** a PySpark DataFrame is produced where each row represents one voxel with columns
   for x, y, z coordinates and T1/FLAIR intensity values, and the row count equals the total
   voxel count of the brain volume.

2. **Given** a subject directory where the T1 file is missing,
   **When** the ingestion module is run,
   **Then** the pipeline raises a descriptive error identifying the missing file and halts
   gracefully without producing partial output.

3. **Given** a T1 and FLAIR scan with mismatched voxel dimensions (different shape arrays),
   **When** the ingestion module is run,
   **Then** the pipeline raises a dimension mismatch error and halts before writing any
   DataFrame partitions.

---

### User Story 2 — Ingest Expert Lesion Masks as Ground Truth (Priority: P2)

For subjects that have a paired expert-annotated binary lesion mask (NIfTI format), the system
loads the mask alongside the MRI sequences and appends the ground truth label to the voxel
DataFrame, enabling downstream DSC evaluation.

**Why this priority**: Required for evaluation runs. The mask is optional per subject (smoke-test
subjects may not have one), so this story must be independently activatable.

**Independent Test**: Can be tested by loading a subject with a known mask and asserting that
the resulting DataFrame contains a `label` column with only 0 and 1 values, and that the
positive voxel count (label=1) matches the mask's non-zero voxel count.

**Acceptance Scenarios**:

1. **Given** a subject directory with T1, FLAIR, and a binary lesion mask,
   **When** the ingestion module is run,
   **Then** the output DataFrame includes a `label` column containing only integer values 0 or 1,
   and the count of rows where `label=1` equals the non-zero voxel count in the mask file.

2. **Given** a subject directory with T1 and FLAIR but no mask file,
   **When** the ingestion module is run without requiring a mask,
   **Then** the pipeline succeeds and the output DataFrame omits the `label` column, with no
   error raised.

3. **Given** a mask file that is not binary (contains values other than 0 and 1),
   **When** the ingestion module is run,
   **Then** the pipeline raises a validation error describing the non-binary values found.

---

### User Story 3 — Partition-Aware Distributed Loading Across Worker Nodes (Priority: P3)

The ingestion layer reads data from the storage volume in a partition-aware manner so that
the initial I/O operation distributes data across worker nodes without bottlenecking the driver.
The partition count is explicitly configurable to respect per-worker memory limits.

**Why this priority**: Essential for CHPC deployment at scale; less critical for local smoke
testing where only one subject is loaded. The P1 and P2 stories must be solid before this
distribution concern is addressed.

**Independent Test**: Can be verified by asserting that the output DataFrame's number of
partitions matches the configured partition count and that no single partition exceeds the
per-worker memory budget derived from voxel count and bytes-per-row.

**Acceptance Scenarios**:

1. **Given** a configuration specifying a target partition count N,
   **When** the ingestion module loads a subject's NIfTI files,
   **Then** the resulting DataFrame has exactly N partitions and each partition contains
   approximately equal row counts (within 10% variance).

2. **Given** no explicit partition count in configuration,
   **When** the ingestion module loads data,
   **Then** the system uses a default partition count calculated from the voxel count and
   a 16 GB per-worker memory ceiling, and logs the derived value.

---

### Edge Cases

- What happens when a NIfTI file is corrupt or truncated mid-read?
  The pipeline raises an I/O error with the file path and halts; no partial DataFrame is emitted.
- What happens when the storage volume is not mounted or the path does not exist?
  The pipeline raises a storage access error at startup before attempting any file reads.
- What happens when a subject has additional scan types beyond T1 and FLAIR (e.g., T2)?
  The ingestion module loads only T1 and FLAIR; additional files are ignored without error.
- What happens when NIfTI files use different affine transforms (different spatial registration)?
  The pipeline detects affine mismatch and raises a registration error; unregistered scans are
  not ingested.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST programmatically ingest paired T1-weighted and FLAIR MRI sequences
  in NIfTI format (.nii or .nii.gz) given a subject directory path.
- **FR-002**: The system MUST validate that the T1 and FLAIR volumes have matching spatial
  dimensions before constructing the output DataFrame.
- **FR-003**: The system MUST flatten the 3D voxel arrays into a 2D tabular structure where
  each row represents one (x, y, z) voxel with associated intensity values.
- **FR-004**: The system MUST optionally ingest a binary expert lesion mask and append a `label`
  column (0 or 1) to the output DataFrame when a mask file is present.
- **FR-005**: The system MUST validate that any ingested mask is strictly binary (only values 0
  and 1); non-binary masks MUST cause a validation error and halt ingestion.
- **FR-006**: The system MUST partition the output DataFrame using an explicit, configurable
  partition count to prevent driver-node bottlenecks.
- **FR-007**: The system MUST log the resolved partition count and voxel count for every
  ingestion run to support benchmarking and debugging.
- **FR-008**: The system MUST fail fast with descriptive error messages when required input
  files are missing, corrupt, or dimensionally mismatched — no silent partial outputs.
- **FR-009**: The system MUST support batch ingestion of multiple subjects from a manifest file
  (list of subject directory paths), processing each subject independently.

### Key Entities

- **Subject**: A unit of data corresponding to one MRI acquisition — identified by a directory
  path containing at minimum a T1 file and a FLAIR file, and optionally a lesion mask.
- **Voxel Record**: One row in the output DataFrame — attributes: subject ID, x, y, z
  coordinates, T1 intensity, FLAIR intensity, optional label (0/1).
- **Ingestion Manifest**: A file listing subject directory paths to be processed in batch mode.
- **Partition Plan**: The derived or configured partition count and strategy applied to the
  output DataFrame.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The ingestion module successfully loads the local `datasets/` smoke-test subject
  (T1_RMS.nii.gz + FLAIR.nii.gz) and produces a non-empty DataFrame with the correct voxel
  schema in under 60 seconds on the local machine.
- **SC-002**: When ingesting a subject with a lesion mask, 100% of mask voxel values are
  correctly reflected in the `label` column with zero label-assignment errors.
- **SC-003**: The output DataFrame partition count matches the configured value (or the
  auto-derived default) in 100% of runs.
- **SC-004**: Ingestion of a corrupt or missing file produces a descriptive error message
  and zero partial DataFrame output in 100% of failure cases tested.
- **SC-005**: The ingestion module can process a batch of 10 subjects sequentially without
  exceeding available memory on the local machine during smoke testing.

## Assumptions

- NIfTI files are already co-registered (same affine transform) before reaching the ingestion
  layer; spatial registration is a pre-condition, not a responsibility of this module.
- During Phase 1 (local smoke testing), the `datasets/` directory contains scans without expert
  masks; mask ingestion will be validated once Kaggle data is available in Phase 2.
- Subject directory structure follows a flat layout: all NIfTI files for a subject reside
  directly in the subject folder (no nested subdirectories required).
- The pipeline runs in a Python 3.9 / PySpark 3.5 environment; no additional NIfTI parsing
  libraries beyond nibabel are assumed to be unavailable.
- The ingestion module does not perform skull-stripping or pre-processing; it loads raw NIfTI
  arrays as-is and delegates stripping to the downstream containerized HD-BET step.
- Mobile or web interfaces are out of scope; all invocation is via Python API or CLI.
