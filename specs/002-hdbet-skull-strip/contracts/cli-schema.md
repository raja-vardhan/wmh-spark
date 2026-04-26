# CLI Contract: skull_strip module

**Module**: `wmh_spark.preprocessing.skull_strip`
**Invocation**: `python -m wmh_spark.preprocessing.skull_strip [OPTIONS]`

## Arguments

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `--input-dir PATH` | Path | Yes (batch mode) | Directory containing raw T1 and FLAIR `.nii.gz` files |
| `--output-dir PATH` | Path | Yes | Directory where skull-stripped outputs are written |
| `--manifest PATH` | Path | No | Parquet manifest file (alternative to `--input-dir`; uses `SubjectRecord` schema) |
| `--subject SUBJECT_ID` | str | No | Process a single subject only (single-scan reprocessing mode) |
| `--smoke-test` | flag | No | Activates Phase 1 mode: skips DSC gate, applies structural assertions only |
| `--docker-image TAG` | str | No | Override Docker image tag (default: `wmh-spark/hdbet:2.0.0`) |
| `--benchmark-out PATH` | Path | No | Override benchmark JSON output path (default: `data/output/metrics/skull_strip_benchmark.json`) |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All scans processed; zero errors |
| `1` | One or more scans rejected by quality gate (batch partially accepted) |
| `2` | Fatal error (Docker unavailable, no input found, output directory unwritable) |

## Output Structure

```
<output-dir>/
├── <subject_id>/
│   ├── T1_RMS_bet.nii.gz          # Skull-stripped T1
│   ├── T1_RMS_bet_mask.nii.gz     # T1 brain mask (binary)
│   ├── FLAIR_bet.nii.gz           # Skull-stripped FLAIR
│   └── FLAIR_bet_mask.nii.gz      # FLAIR brain mask (binary)
└── ...

data/output/metrics/
├── skull_strip_<run_id>.jsonl     # Per-scan log (ProcessingLogEntry)
└── skull_strip_benchmark.json     # Batch-level timing summary
```

## Python API

```python
from wmh_spark.preprocessing.skull_strip import SkullStripper

stripper = SkullStripper(
    output_dir="data/work/skull_stripped",
    docker_image="wmh-spark/hdbet:2.0.0",
    smoke_test=False,
)

result = stripper.process_pair(
    subject_id="sub-001",
    t1_path="datasets/T1_RMS.nii.gz",
    flair_path="datasets/FLAIR.nii.gz",
)
# result: SkullStrippedOutput
```
