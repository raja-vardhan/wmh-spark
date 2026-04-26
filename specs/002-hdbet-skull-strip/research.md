# Research: HD-BET Skull-Stripping Preprocessing Module

**Branch**: `002-hdbet-skull-strip` | **Date**: 2026-04-26

## Decision 1: Local HD-BET Dependency Strategy

**Decision**: Run HD-BET locally through the project Python environment, with `hd-bet==2.0.0`, `torch==2.4.0+cpu`, and `torchvision==0.19.0+cpu` pinned in `requirements.txt`.

**Rationale**: The project now runs local HD-BET directly and no longer requires Docker daemon access. Pinning HD-BET and the PyTorch CPU wheel pair keeps smoke-test behavior reproducible while avoiding container startup and Docker socket failures.

**Alternatives considered**:
- Use a project-owned Docker image — rejected for the current implementation: Docker daemon access is not consistently available in local environments.
- Use a community HD-BET image — rejected: unverified provenance and no guaranteed version pin.

**Pinned versions**: `hd-bet==2.0.0`, `torch==2.4.0+cpu`, `torchvision==0.19.0+cpu`

---

## Decision 2: Local HD-BET Execution from Python

**Decision**: Use Python's `subprocess` module to invoke the local `hd-bet` CLI.

**Rationale**: `subprocess` is stdlib, adds no dependencies, and is sufficient for a single-command HD-BET invocation. Keeping the runner local avoids Docker daemon/socket requirements and still lets the module capture stdout/stderr programmatically.

**Command pattern**:
```
hd-bet -i <scan>.nii.gz -o <output-base>.nii.gz -device cpu --save_bet_mask --disable_tta
```

**Alternatives considered**:
- Shell script wrapper — rejected: harder to test, harder to capture stdout/stderr programmatically.

---

## Decision 3: DSC Quality Gate Without Expert Masks (Smoke-Test Mode)

**Decision**: In smoke-test mode (Phase 1, local `datasets/`), skip the DSC gate entirely. Assert structural correctness instead: brain mask is binary (0/1 values only), skull-stripped volume has the same shape and affine as the input, and the brain mask covers a plausible fraction of the input volume (5–95% foreground voxels).

**Rationale**: The local smoke-test dataset (`datasets/`) has no paired expert masks. Constitution Principle IV explicitly grants a smoke-test exemption from the DSC gate. During Phase 2 (Kaggle full dataset), the DSC gate is mandatory.

**Smoke-test assertions** (replaces DSC check):
1. Brain mask values are strictly binary (0 or 1).
2. Skull-stripped volume shape and affine match the raw input.
3. Brain mask foreground fraction is between 5% and 95% (detects degenerate all-zero or all-one masks).
4. Skull-stripped output file is non-empty and loadable.

**Alternatives considered**:
- Use a simple threshold-based proxy DSC — rejected: requires a reference volume to compute any DSC variant; no reference exists in smoke-test data.

---

## Decision 4: Output File Naming Convention

**Decision**: Derive output names from the input stem:
- Skull-stripped volume: `{stem}_bet.nii.gz`
- Brain mask: `{stem}_bet_mask.nii.gz`

where `{stem}` is the input filename without `.nii.gz` or `.nii`.

**Examples**:
- `T1_RMS.nii.gz` → `T1_RMS_bet.nii.gz`, `T1_RMS_bet_mask.nii.gz`
- `FLAIR.nii.gz` → `FLAIR_bet.nii.gz`, `FLAIR_bet_mask.nii.gz`

**Rationale**: Follows HD-BET's own default naming convention, which makes outputs recognisable and consistent with documentation and community usage.

---

## Decision 5: Integration Point with Existing Pipeline

**Decision**: The skull-stripping module sits upstream of `io_utils.build_voxel_dataframe()`. It operates on raw NIfTI files and writes skull-stripped NIfTI files to a configurable output directory. The existing `SubjectRecord` schema (`t1_path`, `flair_path`) is updated by the skull-stripping stage to point to the stripped output paths before manifest ingestion.

**Rationale**: The existing `validate_subject_volumes()` in `io_utils.py` already enforces co-registration (shape + affine match) before voxel flattening. Skull-stripped volumes pass through the same validation without changes. No modifications to `io_utils.py` are required.

**Flow**:
```
Raw NIfTI files
    ↓  skull_strip.py (local HD-BET)
Stripped NIfTI files + masks
    ↓  quality_gate.py (DSC or smoke-test assertions)
Accepted stripped files
    ↓  io_utils.build_voxel_dataframe()
PySpark DataFrame
```

---

## Decision 6: Per-Scan Timing and Benchmark Logging

**Decision**: Log per-scan wall-clock time to a structured JSON benchmark file, consistent with the existing benchmark pattern in `wmh_spark/benchmark.py`. Per-batch summary (total scans, accepted, rejected, total time) is also logged.

**Rationale**: Constitution Principle V requires every pipeline stage to log per-subject processing time and throughput. The skull-stripping stage is the first per-subject stage and sets the baseline.

**Benchmark output location**: `data/output/metrics/skull_strip_benchmark.json`
