# Tasks: HD-BET Skull-Stripping Preprocessing Module

**Input**: Design documents from `specs/002-hdbet-skull-strip/`
**Prerequisites**: plan.md ✅ spec.md ✅ research.md ✅ data-model.md ✅ contracts/ ✅ quickstart.md ✅

**Tests**: Included — TDD required by Constitution Principle VI (tests must be written and confirmed failing before implementation begins).

**Organization**: Tasks grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no shared dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Paths are relative to `wmh-spark/` (the inner project directory)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Local HD-BET dependency definition and package skeleton. No logic.

- [x] T001 Add local HD-BET dependencies — pin `hd-bet==2.0.0`, `torch==2.4.0+cpu`, and `torchvision==0.19.0+cpu`
- [x] T002 [P] Create `src/wmh_spark/preprocessing/__init__.py` (empty, marks package)
- [x] T003 [P] Create `tests/unit/preprocessing/__init__.py` and `tests/integration/preprocessing/__init__.py` (empty init files)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Data classes and test scaffolding that ALL user story phases depend on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T004 Define `RawScan`, `ScanPair`, `SkullStrippedOutput`, `ProcessingLogEntry` dataclasses in `src/wmh_spark/preprocessing/skull_strip.py` — data definitions only, no logic; fields per `data-model.md`
- [x] T005 [P] Write import-level unit test in `tests/unit/preprocessing/test_skull_strip.py` that instantiates each dataclass with valid fields and asserts field types — confirm test passes (structural smoke, not behaviour)
- [x] T006 [P] Write failing import test in `tests/unit/preprocessing/test_quality_gate.py` — `from wmh_spark.preprocessing import quality_gate` — expected to raise ImportError until T019 creates the module; documents TDD scaffold is in place

**Checkpoint**: Dataclasses importable; test scaffolds in place; user story phases may begin.

---

## Phase 3: User Story 1 — Batch Skull-Strip Raw MRI Scan Pairs (Priority: P1) 🎯 MVP

**Goal**: Given a directory of raw T1/FLAIR NIfTI pairs, run local HD-BET on each and write skull-stripped volumes + brain masks to the output directory. Per-scan timing is logged.

**Independent Test**: Run `pytest tests/unit/preprocessing/test_skull_strip.py -v` and `pytest tests/integration/preprocessing/test_skull_strip_e2e.py::test_smoke_batch -v` — all pass with local HD-BET available.

### Tests for User Story 1 ⚠️ Write and confirm FAILING before T010

- [x] T007 [P] [US1] Write failing unit test for `HDBETRunner._build_command()` in `tests/unit/preprocessing/test_skull_strip.py` — asserts local executable, device flag, mask preservation, and output path suffix `_bet.nii.gz`
- [x] T008 [P] [US1] Write failing unit test for `_discover_pairs()` in `tests/unit/preprocessing/test_skull_strip.py` — mock directory with T1/FLAIR files; assert returns list of `ScanPair` with correct subject_id and paths
- [x] T009 [P] [US1] Write failing unit test for `SkullStripper.process_pair()` in `tests/unit/preprocessing/test_skull_strip.py` — mock `HDBETRunner.run()` to succeed; assert `SkullStrippedOutput` fields are populated and `elapsed_seconds` > 0
- [x] T010 [US1] Write failing integration test `test_smoke_batch` in `tests/integration/preprocessing/test_skull_strip_e2e.py` — calls `SkullStripper.process_batch()` on `../../datasets/` in smoke-test mode; asserts output `_bet.nii.gz` and `_bet_mask.nii.gz` files exist for T1 and FLAIR

### Implementation for User Story 1

- [x] T011 [US1] Implement `HDBETRunner` class in `src/wmh_spark/preprocessing/skull_strip.py` — `_build_command()` constructs a local `hd-bet` command; `run()` calls subprocess, captures stdout/stderr, raises `HDBETExecutionError` on non-zero exit
- [x] T012 [US1] Implement `_discover_pairs()` in `src/wmh_spark/preprocessing/skull_strip.py` — scans input directory for `*T1*.nii*` and `*FLAIR*.nii*` files; groups by subject_id; returns `List[ScanPair]`; raises `ValueError` if no pairs found
- [x] T013 [US1] Implement `SkullStripper.process_pair()` in `src/wmh_spark/preprocessing/skull_strip.py` — orchestrates local HD-BET run for T1 then FLAIR; derives output paths as `{stem}_bet.nii.gz` / `{stem}_bet_mask.nii.gz`; records elapsed time; returns `SkullStrippedOutput`
- [x] T014 [US1] Implement `SkullStripper.process_batch()` in `src/wmh_spark/preprocessing/skull_strip.py` — iterates `ScanPair` list; calls `process_pair()` per pair; catches per-pair exceptions and sets `status="error"` without halting the batch
- [x] T015 [US1] Implement `ProcessingLogEntry` writing in `src/wmh_spark/preprocessing/skull_strip.py` — append one JSON line per scan pair to `data/output/metrics/skull_strip_<run_id>.jsonl`; write batch-level benchmark JSON to `data/output/metrics/skull_strip_benchmark.json` at end of `process_batch()` with fields: `run_id`, `total_pairs`, `accepted`, `rejected`, `errors`, `total_elapsed_seconds`, `subjects_per_hour` (= `total_pairs / total_elapsed_seconds * 3600`) — Constitution Principle V requires throughput logging

**Checkpoint**: `test_smoke_batch` passes. US1 is fully functional and independently testable.

---

## Phase 4: User Story 2 — Quality-Gate Validation Before Downstream Handoff (Priority: P2)

**Goal**: After skull-stripping, validate each output against the DSC threshold (Phase 2) or structural assertions (Phase 1 smoke). Failing pairs are quarantined and excluded from downstream ingestion.

**Independent Test**: Run `pytest tests/unit/preprocessing/test_quality_gate.py -v` — inject a mock zero-mask (DSC ≈ 0) and confirm `status="rejected"` with the pair sent to quarantine.

### Tests for User Story 2 ⚠️ Write and confirm FAILING before T019

- [x] T016 [P] [US2] Write failing unit test for `QualityGate.check_dsc()` in `tests/unit/preprocessing/test_quality_gate.py` — provide synthetic mask pair with known DSC; assert returns `True` when DSC > 0.85 and `False` when ≤ 0.85
- [x] T017 [P] [US2] Write failing unit tests for `QualityGate.assert_structural()` in `tests/unit/preprocessing/test_quality_gate.py` — test four cases: valid binary mask passes; non-binary values fail; shape mismatch fails; foreground fraction 0% fails; foreground fraction 100% fails
- [x] T018 [P] [US2] Write failing unit test for pair-level atomic rejection in `tests/unit/preprocessing/test_quality_gate.py` — mock one scan failing gate; assert both scans in pair are quarantined and `status="rejected"`

### Implementation for User Story 2

- [x] T019 [P] [US2] Implement `QualityGate.assert_structural()` in `src/wmh_spark/preprocessing/quality_gate.py` — loads NIfTI mask via `io_utils.load_mask()`; asserts binary values; asserts output shape == input shape and affines match; asserts foreground fraction in [0.05, 0.95]; raises `QualityGateError` with message on failure
- [x] T020 [P] [US2] Implement `QualityGate.check_dsc()` in `src/wmh_spark/preprocessing/quality_gate.py` — computes `(2 * intersection) / (sum_pred + sum_ref)` between brain mask and expert reference mask; returns DSC float; raises `QualityGateError` if DSC ≤ 0.85
- [x] T021 [US2] Integrate `QualityGate` into `SkullStripper.process_pair()` in `src/wmh_spark/preprocessing/skull_strip.py` — after local HD-BET run: call `assert_structural()` in smoke-test mode or `check_dsc()` in Phase 2 mode; set `SkullStrippedOutput.status` to `"accepted"` / `"rejected"` / `"error"` accordingly
- [x] T022 [US2] Implement quarantine directory logic in `src/wmh_spark/preprocessing/skull_strip.py` — rejected and errored pairs are moved to `<output_dir>/quarantine/<subject_id>/`; accepted pairs remain in `<output_dir>/<subject_id>/`

**Checkpoint**: Injecting a zero-brain-mask causes `status="rejected"` and quarantine. Valid outputs reach `status="accepted"`. US2 independently testable.

---

## Phase 5: User Story 3 — Single-Scan Reprocessing for Debugging (Priority: P3)

**Goal**: A pipeline maintainer can reprocess a single T1/FLAIR pair by subject ID without affecting other outputs. Re-running overwrites previous output and updates the log.

**Independent Test**: Run `pytest tests/integration/preprocessing/test_skull_strip_e2e.py::test_single_subject_reprocess -v` — confirm only target pair output changes; sibling pairs are unchanged.

### Tests for User Story 3 ⚠️ Write and confirm FAILING before T025

- [x] T023 [P] [US3] Write failing unit test for `subject` filter in `tests/unit/preprocessing/test_skull_strip.py` — provide three `ScanPair` objects; call `process_batch(subject="sub-002")`; mock `process_pair()`; assert it is called exactly once with `subject_id="sub-002"`
- [x] T024 [P] [US3] Write failing integration test `test_single_subject_reprocess` in `tests/integration/preprocessing/test_skull_strip_e2e.py` — run full batch first; modify one output file; reprocess that subject only; assert file is restored; assert sibling subject output timestamps are unchanged

### Implementation for User Story 3

- [x] T025 [US3] Add `subject: Optional[str] = None` parameter to `SkullStripper.process_batch()` in `src/wmh_spark/preprocessing/skull_strip.py` — when set, filter `discovered_pairs` to the matching subject_id before processing; raise `ValueError` if subject not found in batch
- [x] T026 [US3] Ensure overwrite behaviour in `SkullStripper.process_pair()` in `src/wmh_spark/preprocessing/skull_strip.py` — output directory for subject is cleared before writing new outputs; new `ProcessingLogEntry` appended to existing log (no deduplication needed per spec)

**Checkpoint**: All three user stories independently testable. Full pytest suite passes.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: CLI entry point, manifest input mode, missing coverage tasks, and end-to-end validation.

- [x] T027 [P] Implement `__main__` CLI entry point with argparse in `src/wmh_spark/preprocessing/skull_strip.py` — flags: `--input-dir`, `--output-dir`, `--manifest`, `--subject`, `--smoke-test`, `--hdbet-bin`, `--device`, `--benchmark-out`; exit codes 0/1/2 per `contracts/cli-schema.md`
- [x] T028 [P] Add manifest-based input mode to `SkullStripper` in `src/wmh_spark/preprocessing/skull_strip.py` — reads Parquet manifest using `io_utils.SubjectRecord` schema; constructs `ScanPair` list from `t1_path` / `flair_path` fields; delegates to `process_batch()`
- [x] T031 [P] Write failing unit test for manifest input mode in `tests/unit/preprocessing/test_skull_strip.py` — mock `pandas.read_parquet()` returning two `SubjectRecord`-shaped rows; assert `SkullStripper` constructs the correct `ScanPair` list and calls `process_batch()`; covers FR-001 manifest path (analysis finding C3)
- [x] T032 [P] Write unit test for deterministic output in `tests/unit/preprocessing/test_skull_strip.py` — call `process_pair()` twice on the same input with mocked `HDBETRunner.run()`; assert SHA-256 checksums of both output files are identical; covers FR-008 / SC-004 (analysis finding C1)
- [x] T033 [P] Write unit test for 60% voxel reduction in `tests/unit/preprocessing/test_quality_gate.py` — create synthetic raw volume and a stripped version with > 60% voxels zeroed; assert `count_nonzero(stripped) / total_voxels(raw) < 0.40`; covers SC-003 / US1 Acceptance Scenario 2 (analysis finding C2)
- [x] T029 Run full unit + integration test suite and confirm zero failures: `cd wmh-spark && python -m pytest tests/unit/preprocessing/ tests/integration/preprocessing/ -v`
- [ ] T030 Execute smoke-test end-to-end validation per `specs/002-hdbet-skull-strip/quickstart.md` Steps 2–4 against `datasets/`; confirm JSONL log and benchmark JSON (with `subjects_per_hour` field) are produced

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 — BLOCKS all user story phases
- **US1 (Phase 3)**: Depends on Phase 2 — no dependency on US2 or US3
- **US2 (Phase 4)**: Depends on Phase 2 — no dependency on US1 (quality gate is isolated in `quality_gate.py`); integrates into skull_strip.py in T021
- **US3 (Phase 5)**: Depends on Phase 2 — integrates lightly with US1 (process_batch filter)
- **Polish (Phase 6)**: Depends on Phase 3 + Phase 4 + Phase 5

### User Story Dependencies

- **US1 (P1)**: Starts after Phase 2 — fully independent
- **US2 (P2)**: Starts after Phase 2 — `quality_gate.py` is independent; integration into `skull_strip.py` (T021) requires T013 to be done
- **US3 (P3)**: Starts after Phase 2 — requires T014 (`process_batch`) to be done before T025

### Within Each User Story

1. Write tests → confirm they FAIL
2. Implement → confirm tests PASS
3. Checkpoint before moving to next story

---

## Parallel Opportunities

### Phase 1

```
T001 (local HD-BET deps)  ||  T002 (preprocessing/__init__.py)  ||  T003 (test __init__ files)
```

### Phase 2

```
T004 (dataclasses)  →  T005 (dataclass tests) [P]  ||  T006 (quality_gate import test) [P]
```

### Phase 3 — Tests (write in parallel, all must FAIL first)

```
T007 (HDBETRunner command test) [P]
T008 (pair discovery test) [P]
T009 (process_pair unit test) [P]
T010 (integration smoke test) — serial after T007-T009 fixtures set up
```

### Phase 3 — Implementation

```
T011 (HDBETRunner) [P]  ||  T012 (_discover_pairs) [P]
    ↓
T013 (process_pair)  →  T014 (process_batch)  →  T015 (logging)
```

### Phase 4 — Tests (write in parallel)

```
T016 (DSC test) [P]  ||  T017 (structural assertions test) [P]  ||  T018 (pair rejection test) [P]
```

### Phase 4 — Implementation

```
T019 (assert_structural) [P]  ||  T020 (check_dsc) [P]
    ↓
T021 (integrate into process_pair)  →  T022 (quarantine logic)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: `test_smoke_batch` passes; output files inspectable
5. Demo/review before proceeding to US2

### Incremental Delivery

1. Phases 1–2 → package skeleton ready
2. Phase 3 (US1) → batch skull-stripping works end-to-end
3. Phase 4 (US2) → quality gate active; quarantine working
4. Phase 5 (US3) → single-scan reprocessing available
5. Phase 6 → CLI polished; manifest mode; full validation

---

## Notes

- `[P]` tasks touch different files — safe to parallelize
- Tests must FAIL before implementation per Constitution Principle VI
- Constitution Principle II: HD-BET dependency versions must stay pinned
- Constitution Principle IV smoke-test exemption: `--smoke-test` flag bypasses DSC gate; structural assertions apply instead
- Constitution Principle V: benchmark JSON must be produced on every run, even smoke-test
- All file paths relative to `wmh-spark/` (inner project directory containing `src/` and `tests/`)
