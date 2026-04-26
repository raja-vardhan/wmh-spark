# Quickstart: HD-BET Skull-Stripping Module

## Prerequisites

- Python 3.11 environment with project dependencies installed (`pip install -r requirements.txt && pip install -e .` in `wmh-spark/`)
- Local HD-BET CLI installed in the project environment (`hd-bet --help` when activated, or `.venv/bin/hd-bet --help`)

## Step 1: Install Local HD-BET Dependencies

```bash
cd wmh-spark
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify the CLI:
```bash
hd-bet --help
```

The skull-strip module defaults to the `hd-bet` executable installed next to
the Python interpreter running it, so `.venv/bin/python -m ...` works even if
the virtualenv shell activation was skipped.

## Step 2: Run Skull-Stripping on Smoke-Test Dataset (Phase 1)

```bash
cd wmh-spark
python -m wmh_spark.preprocessing.skull_strip \
    --input-dir ../datasets/ \
    --output-dir data/work/skull_stripped/ \
    --smoke-test \
    --device cpu
```

Expected output:
- `data/work/skull_stripped/<subject_id>/T1_RMS_bet.nii.gz`
- `data/work/skull_stripped/<subject_id>/FLAIR_bet.nii.gz`
- `data/work/skull_stripped/<subject_id>/T1_RMS_bet_mask.nii.gz`
- `data/work/skull_stripped/<subject_id>/FLAIR_bet_mask.nii.gz`
- `data/output/metrics/skull_strip_<run_id>.jsonl`

## Step 3: Verify Structural Assertions (Smoke Mode)

```bash
python -m pytest tests/unit/preprocessing/ -v
python -m pytest tests/integration/preprocessing/ -v
```

All tests must pass before proceeding to Phase 2.

## Step 4: Run on Kaggle WMH Dataset (Phase 2 — CHPC)

```bash
python -m wmh_spark.preprocessing.skull_strip \
    --manifest data/manifest.parquet \
    --output-dir data/work/skull_stripped/ \
    --device cpu
```

The DSC quality gate (> 0.85) is active by default in Phase 2 mode. Rejected scans are written to `data/work/skull_stripped/quarantine/`.

## Step 5: Reprocess a Single Subject

```bash
python -m wmh_spark.preprocessing.skull_strip \
    --input-dir data/raw/ \
    --output-dir data/work/skull_stripped/ \
    --subject sub-042
```

## Benchmark Output

After any run, check the timing summary:
```bash
cat data/output/metrics/skull_strip_benchmark.json
```

Fields: `run_id`, `total_pairs`, `accepted`, `rejected`, `errors`, `total_elapsed_seconds`, `subjects_per_hour`.
