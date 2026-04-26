# Implementation Plan: HD-BET Skull-Stripping Preprocessing Module

**Branch**: `002-hdbet-skull-strip` | **Date**: 2026-04-26 | **Spec**: [spec.md](spec.md)

## Summary

Implement a skull-stripping preprocessing module that runs local HD-BET on all raw T1 and FLAIR NIfTI scan pairs, validates output quality (DSC > 0.85 in Phase 2; structural assertions in Phase 1 smoke-test mode), and forwards accepted skull-stripped volumes to the existing `build_voxel_dataframe()` ingestion path. The module integrates with the existing `wmh_spark` package structure and follows TDD with tests written before implementation.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: HD-BET 2.0.0, PyTorch CPU wheels, NiBabel (existing), subprocess (stdlib), pytest (existing)  
**Storage**: NIfTI files on local filesystem (Phase 1) / CHPC network volume (Phase 2); JSON Lines benchmark logs  
**Testing**: pytest; unit tests for local HD-BET runner, quality gate, output naming; integration test against local `datasets/`  
**Target Platform**: Linux (local dev + CHPC cluster nodes with local HD-BET installed)  
**Project Type**: Pipeline preprocessing module (sits between raw data and PySpark ingestion)  
**Performance Goals**: Process one scan pair in < 5 min with GPU-capable local HD-BET; deterministic output (bit-identical across runs on same dependency versions)  
**Constraints**: HD-BET and PyTorch versions MUST be pinned; DSC > 0.85 gate mandatory in Phase 2; TDD required  
**Scale/Scope**: 4 scan pairs (smoke), ~hundreds of subjects (Kaggle Phase 2)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Distributed-First | **COMPLIANT** (justified exception) | Skull-stripping is a per-file NIfTI operation that must precede the 3D→2D voxel flattening boundary. It cannot be expressed as a DataFrame transformation. Accepted preprocessing boundary — outputs feed directly into `build_voxel_dataframe()`. |
| II. Reproducible Pre-Processing | **COMPLIANT** | HD-BET runs through pinned local dependencies. Execution is orchestrated by the module, not invoked manually. |
| III. Memory-Safe Partitioning | **N/A** | No PySpark operations in this module. Partition strategy applies downstream in `build_voxel_dataframe()`. |
| IV. Accuracy-Gated Output | **COMPLIANT** | DSC > 0.85 gate enforced in Phase 2. Smoke-test exemption applied in Phase 1 with structural assertions per Constitution. |
| V. Scalability & Benchmarking | **COMPLIANT** | Per-scan elapsed time and batch-level throughput logged to JSONL + benchmark JSON on every run. |
| VI. Test-Driven Correctness | **COMPLIANT** | Tests written before implementation (Red-Green-Refactor). Unit tests cover local runner, quality gate, naming logic. Integration test covers end-to-end against local `datasets/`. |

**Post-design re-check**: No violations introduced in Phase 1 design. No Complexity Tracking entries required.

## Project Structure

### Documentation (this feature)

```text
specs/002-hdbet-skull-strip/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── cli-schema.md    # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit-tasks)
```

### Source Code (repository root)

```text
wmh-spark/
├── src/wmh_spark/
│   └── preprocessing/
│       ├── __init__.py
│       ├── skull_strip.py                # SkullStripper class + CLI entry point
│       └── quality_gate.py              # DSC gate + smoke-test structural assertions
│
└── tests/
    ├── unit/
    │   └── preprocessing/
    │       ├── test_skull_strip.py       # Local HD-BET runner, output naming, batch logic
    │       └── test_quality_gate.py      # DSC gate, structural assertions, edge cases
    └── integration/
        └── preprocessing/
            └── test_skull_strip_e2e.py   # End-to-end: smoke-test dataset, single-pair mode
```

**Structure Decision**: Single-project layout extending the existing `wmh_spark` package. `preprocessing/` is a new sub-package. Tests mirror the `unit/` + `integration/` split already established in the project.
