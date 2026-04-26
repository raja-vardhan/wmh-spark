---
description: "Task list for Distributed MRI Data Ingestion & I/O Layer"
---

# Tasks: Distributed MRI Data Ingestion & I/O Layer

**Input**: Design documents from `specs/001-mri-data-ingestion/`
**Prerequisites**: spec.md ✅ | plan.md ⚠ not generated (tech stack derived from constitution + git history)
**Tech stack**: Python 3.9, PySpark 3.5, nibabel 5.2.1, pytest 8.1.1
**Source root**: `wmh-spark/src/wmh_spark/` | **Tests root**: `wmh-spark/tests/`
**Tests**: Included — constitution Principle VI mandates Red-Green-Refactor (write failing test first)

**Organization**: Tasks grouped by user story. Each story is independently testable.

---

## Phase 1: Setup (Project Scaffold)

**Purpose**: Create the directory layout and dependency files that all stories depend on.

- [ ] T001 Create `wmh-spark/` directory structure: `src/wmh_spark/`, `tests/`, `configs/`, `scripts/`, `docs/`
- [ ] T002 Create `wmh-spark/requirements.txt` with pinned dependencies: pyspark==3.5.1, nibabel==5.2.1, numpy==1.26.4, pyarrow==14.0.2, pyyaml==6.0.1, pytest==8.1.1, pytest-cov==5.0.0
- [ ] T003 [P] Create `wmh-spark/src/wmh_spark/__init__.py` as an empty package init file
- [ ] T004 [P] Create `wmh-spark/configs/local.yaml` with local dev settings: `spark.master: local[4]`, `spark.driver_memory: 8g`, `paths.data_dir: datasets/`

**Checkpoint**: `pip install -r wmh-spark/requirements.txt` succeeds and `python -c "import nibabel, pyspark"` runs without error.

---

## Phase 2: Foundation (Blocking Prerequisites)

**Purpose**: Core primitives and infrastructure that ALL user stories depend on. No story work begins until this phase is complete.

**⚠ CRITICAL**: US1, US2, and US3 all block on this phase.

- [ ] T005 Create `wmh-spark/src/wmh_spark/config.py` with frozen dataclasses `SparkConfig`, `PathsConfig`, `IngestionConfig` (partition_count: int = 0 means auto-derive), and `load_config(path)` / `dump_config(cfg, path)` functions using PyYAML
- [ ] T006 Create `wmh-spark/src/wmh_spark/io_utils.py` with `SubjectRecord` dataclass (subject_id, t1_path, flair_path, gt_mask_path optional), `load_volume(path) -> (ndarray, affine)`, `save_volume(data, affine, path)`, and `load_mask(path) -> ndarray` primitives using nibabel
- [ ] T007 [P] Create `wmh-spark/tests/conftest.py` with pytest fixtures: `spark_session` (session-scoped SparkSession in local mode), `synthetic_volume_shape` (24,32,28), `synthetic_affine` (2mm isotropic), and `synthetic_subject(tmp_path)` that writes paired flair.nii.gz, t1.nii.gz, and wmh_mask.nii.gz synthetic volumes to a temp directory
- [ ] T008 [P] Create `wmh-spark/tests/test_config.py` with tests for `load_config()` round-trip (write YAML, load, assert values), missing key defaults, and `IngestionConfig.partition_count` default value

**Checkpoint**: `pytest wmh-spark/tests/test_config.py` passes. `from wmh_spark.io_utils import load_volume, SubjectRecord` succeeds.

---

## Phase 3: User Story 1 — Load Paired T1 + FLAIR into Spark DataFrame (Priority: P1) 🎯 MVP

**Goal**: Given a subject directory with T1 and FLAIR NIfTI files, produce a Spark DataFrame where each row is one (x, y, z) voxel with columns: `subject_id`, `x`, `y`, `z`, `t1`, `flair`.

**Independent Test**: Run against local `datasets/T1_RMS.nii.gz` + `datasets/FLAIR.nii.gz`. Assert DataFrame schema, non-zero row count, and that row count equals T1 volume's voxel count.

### Tests for User Story 1 (write FIRST — confirm they FAIL before T012/T013)

- [ ] T009 [P] [US1] Write test `test_validate_subject_volumes_mismatch` in `wmh-spark/tests/test_io_utils.py`: create two synthetic volumes with different shapes, call `validate_subject_volumes()`, assert `ValueError` is raised with message containing "dimension mismatch"
- [ ] T010 [P] [US1] Write test `test_validate_subject_volumes_ok` in `wmh-spark/tests/test_io_utils.py`: call `validate_subject_volumes()` with matching shapes, assert no exception raised
- [ ] T011 [P] [US1] Write test `test_build_voxel_dataframe_schema` in `wmh-spark/tests/test_io_utils.py`: call `build_voxel_dataframe()` using `synthetic_subject` fixture and `spark_session`, assert columns `["subject_id","x","y","z","t1","flair"]` exist and row count equals `24*32*28`

### Implementation for User Story 1

- [ ] T012 [US1] Implement `validate_subject_volumes(t1_path, flair_path)` in `wmh-spark/src/wmh_spark/io_utils.py`: load both volumes with nibabel, raise `ValueError` if shapes differ or affines differ by more than 1e-4, return `(t1_data, flair_data, affine)`
- [ ] T013 [US1] Implement `build_voxel_dataframe(spark, subject_id, t1_path, flair_path, partition_count)` in `wmh-spark/src/wmh_spark/io_utils.py`: call `validate_subject_volumes`, use `np.indices` to get (x,y,z) coords, flatten all arrays, create pandas DataFrame, convert to Spark DataFrame via `spark.createDataFrame`, then call `.repartition(partition_count)` — **Phase 1 trade-off**: `createDataFrame` from pandas materializes the full voxel array in driver memory (~320 MB for a 256³ brain); acceptable for local smoke testing but see T032 for the CHPC-scale replacement
- [ ] T014 [US1] Write smoke test script `wmh-spark/scripts/smoke_test_io.py`: load config from `wmh-spark/configs/local.yaml`, start SparkSession, record `start = time.time()`, call `build_voxel_dataframe` on `datasets/T1_RMS.nii.gz` + `datasets/FLAIR.nii.gz`, print schema and `df.count()`, assert `count > 0`, then `assert (time.time() - start) < 60, f"Ingestion exceeded 60s SC-001 threshold"`

**Checkpoint**: `python wmh-spark/scripts/smoke_test_io.py` runs without error. `pytest wmh-spark/tests/test_io_utils.py::test_build_voxel_dataframe_schema` passes.

---

## Phase 4: User Story 2 — Ingest Expert Lesion Mask as Ground Truth (Priority: P2)

**Goal**: Extend `build_voxel_dataframe()` to optionally accept a mask path and append a `label` column (0 or 1). Validate the mask is strictly binary before appending.

**Independent Test**: Use `synthetic_subject` fixture (which includes wmh_mask.nii.gz). Assert `label` column exists, contains only 0/1 values, and the count of `label=1` rows matches the non-zero voxel count in the mask file.

### Tests for User Story 2 (write FIRST — confirm they FAIL before T018/T019)

- [ ] T015 [P] [US2] Write test `test_validate_binary_mask_fails_non_binary` in `wmh-spark/tests/test_io_utils.py`: save a volume with values [0,1,2] using `save_volume`, call `validate_binary_mask()`, assert `ValueError` raised with message containing "non-binary"
- [ ] T016 [P] [US2] Write test `test_build_voxel_dataframe_with_mask` in `wmh-spark/tests/test_io_utils.py`: call `build_voxel_dataframe()` with mask path from `synthetic_subject`, assert `label` column present, assert `df.filter("label not in (0,1)").count() == 0`, assert `df.filter("label=1").count() > 0`
- [ ] T017 [P] [US2] Write test `test_build_voxel_dataframe_without_mask` in `wmh-spark/tests/test_io_utils.py`: call `build_voxel_dataframe()` with `mask_path=None`, assert `label` column is NOT present in the schema

### Implementation for User Story 2

- [ ] T018 [US2] Implement `validate_binary_mask(mask_path)` in `wmh-spark/src/wmh_spark/io_utils.py`: load volume with nibabel, check that `np.unique(data)` is a subset of `{0, 1}`, raise `ValueError` listing the unexpected values if not
- [ ] T019 [US2] Extend `build_voxel_dataframe()` in `wmh-spark/src/wmh_spark/io_utils.py` to accept `mask_path: Optional[str] = None`: when provided, call `validate_binary_mask`, flatten the mask array, and append `label` column (int) to the pandas DataFrame before Spark conversion

**Checkpoint**: `pytest wmh-spark/tests/test_io_utils.py -k "mask"` passes all 3 mask-related tests.

---

## Phase 5: User Story 3 — Partition-Aware Distributed Loading (Priority: P3)

**Goal**: When no partition count is explicitly configured, auto-derive a safe partition count from voxel count and per-worker memory ceiling (16 GB). Log the resolved partition count on every run.

**Independent Test**: Assert `derive_partition_count(voxel_count=1000000, bytes_per_row=28, max_partition_bytes=1<<34)` returns the expected integer. Assert that calling `build_voxel_dataframe()` with `partition_count=0` (auto-derive) results in a DataFrame whose actual partition count equals the derived value.

### Tests for User Story 3 (write FIRST — confirm they FAIL before T023/T024)

- [ ] T020 [P] [US3] Write test `test_derive_partition_count_auto` in `wmh-spark/tests/test_io_utils.py`: call `derive_partition_count(voxel_count=500000, bytes_per_row=28)`, assert result is a positive integer >= 1 and that `500000 * 28 / result <= (16 * 1024**3)`
- [ ] T021 [P] [US3] Write test `test_derive_partition_count_override` in `wmh-spark/tests/test_io_utils.py`: call `derive_partition_count(voxel_count=500000, bytes_per_row=28, override=8)`, assert result equals 8
- [ ] T022 [US3] Write test `test_build_voxel_dataframe_partition_count` in `wmh-spark/tests/test_io_utils.py`: (a) call `build_voxel_dataframe()` with `partition_count=4`, assert `df.rdd.getNumPartitions() == 4`; (b) call `build_voxel_dataframe()` with `partition_count=0` (auto-derive), assert `df.rdd.getNumPartitions() >= 1` (covers US3 Independent Test auto-derive path)

### Implementation for User Story 3

- [ ] T023 [US3] Implement `derive_partition_count(voxel_count, bytes_per_row=28, max_partition_bytes=16*1024**3, override=0)` in `wmh-spark/src/wmh_spark/io_utils.py`: if `override > 0` return it; otherwise compute `ceil(voxel_count * bytes_per_row / max_partition_bytes)`, floor at 1; log the computed value with Python's `logging` module
- [ ] T024 [US3] Wire `derive_partition_count()` into `build_voxel_dataframe()` in `wmh-spark/src/wmh_spark/io_utils.py`: when `partition_count=0`, call `derive_partition_count(voxel_count, override=0)` to resolve the value before calling `.repartition()`

**Checkpoint**: `pytest wmh-spark/tests/test_io_utils.py -k "partition"` passes. Running smoke test with no partition config logs a derived partition count.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Batch manifest support, checksum verification, structured benchmark logging, and documentation.

- [ ] T025 [P] Create `wmh-spark/scripts/make_manifest.py`: scan a root data directory for subject folders (each must contain `t1*.nii*` and `flair*.nii*`), write a `manifest.parquet` file with one `SubjectRecord` row per subject using pyarrow; accept `--data-dir` and `--output` CLI args via argparse
- [ ] T030 [P] Implement `ingest_from_manifest(spark, manifest_path, config)` in `wmh-spark/src/wmh_spark/io_utils.py`: read the Parquet manifest via `spark.read.parquet`, iterate rows (driver-side coordination only), call `build_voxel_dataframe` per subject, return list of (subject_id, DataFrame) tuples; this covers FR-009
- [ ] T031 Write batch OOM smoke test in `wmh-spark/scripts/smoke_test_batch.py`: use `make_manifest.py` to write a 10-entry synthetic manifest (10 copies of the `synthetic_subject` fixture paths), call `ingest_from_manifest`, assert all 10 DataFrames have non-zero row counts; covers SC-005
- [ ] T026a [P] Write test `test_sha256_stable` in `wmh-spark/tests/test_io_utils.py` asserting `sha256_of_file` returns a 64-char hex string and is deterministic across two calls — confirm it FAILS before T026b
- [ ] T026b [P] Implement `sha256_of_file(path)` and `write_checksum_manifest(records, out_path)` in `wmh-spark/src/wmh_spark/io_utils.py` — run T026a after to confirm it passes
- [ ] T027 Create `wmh-spark/src/wmh_spark/benchmark.py` with `log_ingestion_run(subject_id, voxel_count, partition_count, elapsed_seconds, output_path)` that appends a JSON line to a benchmark log file at `paths.log_dir/ingestion_bench.jsonl`
- [ ] T028 [P] Wire `benchmark.log_ingestion_run()` into `build_voxel_dataframe()` (measure wall-clock time from start to `.count()` probe, then log)
- [ ] T029 [P] Update smoke test script `wmh-spark/scripts/smoke_test_io.py` to also call `log_ingestion_run` and print the benchmark JSON line to stdout — **Note**: benchmark logs `elapsed_seconds` only in Phase 1; Phase 2 extension needed for `subjects_per_hour` and 1→4 node scaling factor per Constitution Principle V
- [ ] T032 [P] Add architectural note and TODO in `wmh-spark/src/wmh_spark/io_utils.py` above `build_voxel_dataframe`: document that the `pandas→spark.createDataFrame` pattern is a Phase 1 trade-off and flag a Phase 2 replacement using `rdd.mapPartitions` + per-worker nibabel reads to eliminate driver materialization (resolves Constitution Principle III at CHPC scale)
- [ ] T033 [P] Write test `test_validate_subject_volumes_missing_t1` in `wmh-spark/tests/test_io_utils.py`: call `validate_subject_volumes` with a non-existent T1 path, assert `FileNotFoundError` is raised with the missing path in the message; covers FR-008 + SC-004
- [ ] T034 [P] Write test `test_load_volume_corrupt_file` in `wmh-spark/tests/test_io_utils.py`: write a 10-byte garbage file at `tmp_path/"corrupt.nii.gz"`, call `load_volume`, assert an `IOError` or nibabel-raised exception is caught and re-raised with the file path; covers FR-008 edge case

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundation (Phase 2)**: Depends on Phase 1 ⚠ BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundation complete — tests written before T012/T013
- **US2 (Phase 4)**: Depends on Foundation complete, integrates with US1 implementation
- **US3 (Phase 5)**: Depends on Foundation complete, extends US1 implementation
- **Polish (Phase 6)**: Depends on US1+US2+US3 complete; T030/T031 depend on T013 (build_voxel_dataframe); T033/T034 are independent test tasks that can run after Foundation

### User Story Dependencies

- **US1 (P1)**: Can start immediately after Foundation — no dependency on US2 or US3
- **US2 (P2)**: Can start immediately after Foundation — depends on `build_voxel_dataframe` from US1 (T013)
- **US3 (P3)**: Can start immediately after Foundation — depends on `build_voxel_dataframe` from US1 (T013)

### Within Each Story

- Test tasks (T009–T011, T015–T017, T020–T022) MUST be written first and confirmed FAILING
- Models / primitives before services before integration
- Smoke test (`smoke_test_io.py`) runs after US1 checkpoint to confirm real NIfTI loading works

### Parallel Opportunities

- T003, T004 (Setup) — parallel
- T007, T008 (Foundation) — parallel
- T009, T010, T011 (US1 tests) — all parallel
- T015, T016, T017 (US2 tests) — all parallel
- T020, T021 (US3 tests) — parallel
- T025, T026a, T026b, T028, T029, T030, T032, T033, T034 (Polish) — parallel where marked [P]
- T026a MUST precede T026b (TDD gate)
- T031 depends on T030 (batch ingest loop) being complete

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundation (CRITICAL — blocks all stories)
3. Complete Phase 3: US1 tests (T009–T011) → confirm FAILING → implement (T012–T014)
4. **STOP and VALIDATE**: Run `python wmh-spark/scripts/smoke_test_io.py` against `datasets/`
5. US1 DataFrame output is the input for all downstream pipeline features

### Incremental Delivery

1. Setup + Foundation → scaffold ready
2. US1 → MVP: can load any paired T1+FLAIR into Spark ✅
3. US2 → adds ground-truth labels for evaluation runs ✅
4. US3 → adds memory-safe partitioning for CHPC scale ✅

### Parallel Opportunities (with multiple developers)

- After Foundation complete: Developer A on US1, Developer B on US2 tests (mock `build_voxel_dataframe`)
- US3 can start in parallel once T013 (`build_voxel_dataframe` signature) is committed

---

## Notes

- `[P]` = parallelizable with other `[P]` tasks (different files, no incomplete dependencies)
- `[USN]` = maps task to User Story N for traceability
- Constitution Principle III: all `.repartition()` and `.coalesce()` call sites MUST have a comment documenting the partition count rationale
- Constitution Principle VI: TDD is NON-NEGOTIABLE — run `pytest ... -v` to confirm tests FAIL before writing the implementation
- Smoke test data path: `datasets/T1_RMS.nii.gz` (T1) and `datasets/FLAIR.nii.gz` (FLAIR) — no mask available locally; US2 mask validation uses synthetic fixtures only until Kaggle data arrives
- `bytes_per_row` in `derive_partition_count` = 4 (subject_id ptr) + 4+4+4 (x,y,z int32) + 4+4 (t1,flair float32) + 4 (label int32, when present) ≈ 28 bytes/row
