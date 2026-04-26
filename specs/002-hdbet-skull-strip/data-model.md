# Data Model: HD-BET Skull-Stripping Preprocessing Module

**Branch**: `002-hdbet-skull-strip` | **Date**: 2026-04-26

## Entities

### RawScan

Represents a single unprocessed NIfTI volume as it arrives in the pipeline.

| Field | Type | Constraints |
|-------|------|-------------|
| `path` | `Path` | Must exist; `.nii` or `.nii.gz` |
| `modality` | `Literal["T1", "FLAIR"]` | Required |
| `subject_id` | `str` | Non-empty |

---

### ScanPair

An atomic unit: one T1 and one FLAIR scan belonging to the same subject and session. Treated atomically for quality-gating — if either scan fails, the whole pair is quarantined.

| Field | Type | Constraints |
|-------|------|-------------|
| `subject_id` | `str` | Non-empty; unique within a batch |
| `t1_scan` | `RawScan` | modality == "T1" |
| `flair_scan` | `RawScan` | modality == "FLAIR" |

**Relationships**: `ScanPair` is composed of exactly two `RawScan` instances.

---

### SkullStrippedOutput

The result of running HD-BET on one `ScanPair`.

| Field | Type | Constraints |
|-------|------|-------------|
| `subject_id` | `str` | Matches source `ScanPair.subject_id` |
| `t1_stripped_path` | `Path` | `{stem}_bet.nii.gz` |
| `t1_mask_path` | `Path` | `{stem}_bet_mask.nii.gz` |
| `flair_stripped_path` | `Path` | `{stem}_bet.nii.gz` |
| `flair_mask_path` | `Path` | `{stem}_bet_mask.nii.gz` |
| `t1_dsc` | `Optional[float]` | `[0.0, 1.0]`; `None` in smoke-test mode |
| `flair_dsc` | `Optional[float]` | `[0.0, 1.0]`; `None` in smoke-test mode |
| `status` | `Literal["accepted", "rejected", "error"]` | Required |
| `error_message` | `Optional[str]` | Set when `status == "error"` |

**State transitions**:
```
ScanPair  →  (HD-BET runs)  →  SkullStrippedOutput(status="accepted"|"rejected"|"error")
```
- `accepted`: Both T1 and FLAIR pass quality gate; forwarded to downstream ingestion.
- `rejected`: One or both scans fail DSC gate (Phase 2) or structural assertions (Phase 1).
- `error`: Local HD-BET execution failed or output file is unreadable.

---

### ProcessingLogEntry

One entry per processed `ScanPair`, written to the run's structured log.

| Field | Type | Constraints |
|-------|------|-------------|
| `run_id` | `str` | UUID; one per batch invocation |
| `timestamp` | `str` | ISO-8601 UTC |
| `subject_id` | `str` | Matches `ScanPair.subject_id` |
| `t1_dsc` | `Optional[float]` | DSC value or `null` in smoke mode |
| `flair_dsc` | `Optional[float]` | DSC value or `null` in smoke mode |
| `status` | `str` | `"accepted"` / `"rejected"` / `"error"` |
| `elapsed_seconds` | `float` | Wall-clock time for this pair |
| `error_message` | `Optional[str]` | Populated on error |

**Storage**: JSON Lines (`.jsonl`) at `data/output/metrics/skull_strip_<run_id>.jsonl`.

---

## Data Flow

```
Input directory / manifest
        │
        ▼
[ScanPair discovery]
        │  discovers T1 + FLAIR paths per subject
        ▼
[Local HD-BET execution]
        │  produces skull-stripped NIfTI + binary mask per scan
        ▼
[Quality gate]
        │  Phase 2: DSC > 0.85 against expert mask
        │  Phase 1: structural assertions (binary mask, shape/affine match,
        │            foreground fraction 5–95%)
        ▼
[SkullStrippedOutput]
        │  accepted  →  downstream io_utils.build_voxel_dataframe()
        │  rejected  →  quarantine directory
        │  error     →  quarantine directory
        ▼
[ProcessingLogEntry written to JSONL + benchmark JSON]
```

---

## Integration with Existing Schema

The skull-stripping module updates the `SubjectRecord.t1_path` and `SubjectRecord.flair_path` fields (defined in `io_utils.py`) to point to skull-stripped output paths before the manifest is passed to `build_voxel_dataframe()`. No schema changes to `SubjectRecord` are required.
