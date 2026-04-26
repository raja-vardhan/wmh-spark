# Research: HD-BET Skull-Stripping Preprocessing Module

**Branch**: `002-hdbet-skull-strip` | **Date**: 2026-04-26

## Decision 1: HD-BET Docker Image Strategy

**Decision**: Build a project-owned Docker image (`wmh-spark/hdbet:2.0.0`) from a pinned base, with HD-BET installed via pip at a fixed version.

**Rationale**: There is no official HD-BET Docker Hub image. HD-BET is pip-installable (`hd-bet==2.0.0`). Building a project-owned image from a pinned PyTorch CUDA base guarantees identical tool behaviour across all CHPC nodes and local dev runs, satisfying Constitution Principle II.

**Alternatives considered**:
- Use a community HD-BET Docker image — rejected: unverified provenance, no guaranteed version pin, may not match CHPC node GPU drivers.
- Install HD-BET bare-metal on CHPC — rejected: explicitly prohibited by Constitution Principle II.

**Pinned base**: `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime`
**Image tag**: `wmh-spark/hdbet:2.0.0`
**Dockerfile location**: `docker/hdbet/Dockerfile`

---

## Decision 2: Docker Execution from Python

**Decision**: Use Python's `subprocess` module to invoke `docker run` commands.

**Rationale**: `subprocess` is stdlib, adds no dependencies, and is sufficient for a single-command container invocation. The `docker` Python SDK (docker-py) would add an extra dependency and requires the Docker daemon socket to be accessible via a Python binding — no advantage for this use case.

**Command pattern**:
```
docker run --rm --gpus all \
  -v <input_dir>:/input:ro \
  -v <output_dir>:/output \
  wmh-spark/hdbet:2.0.0 \
  hd_bet -i /input/<scan>.nii.gz -o /output/<scan>_bet.nii.gz
```

**Alternatives considered**:
- docker-py SDK — rejected: extra dependency with no functional advantage for single-command invocation.
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
    ↓  skull_strip.py (Docker HD-BET)
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
