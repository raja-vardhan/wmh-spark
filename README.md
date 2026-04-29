# wmh-spark — Distributed WMH Segmentation Pipeline

A scalable PySpark re-engineering of the legacy MATLAB UBO Detector for quantifying
White Matter Hyperintensities (WMH) on paired T1 / FLAIR MRI volumes. The pipeline
performs HD-BET skull stripping, builds a per-voxel feature DataFrame, trains a
distributed Spark MLlib Random Forest, and post-processes the predictions back into
3D NIfTI masks with Dice and lesion-level reporting.

This document covers the **end-to-end pipeline**: requirements, installation,
dataset layout, and how to run training + inference + reporting in one command.

---

## 1. Requirements

### System

| Requirement | Version | Notes |
|-------------|---------|-------|
| OS          | Linux (Ubuntu/Debian/Fedora/RHEL) or macOS | Tested on Linux 6.x |
| Python      | **3.11.x** (exact minor) | PySpark 3.5.1 driver + worker minor must match |
| Java (JDK)  | **17** | Required by Spark 3.5 |
| Disk        | ~10 GB free | HD-BET model weights + cached outputs |
| RAM         | 16 GB minimum (32 GB recommended) | Spark driver default is 6 GB |
| GPU         | Optional (CUDA / ROCm / Apple MPS) | Speeds up HD-BET; CPU works |

### Python dependencies (pinned)

Installed automatically by [wmh-spark/scripts/setup.sh](wmh-spark/scripts/setup.sh)
from [wmh-spark/requirements.txt](wmh-spark/requirements.txt):

- `pyspark==3.5.1`, `py4j==0.10.9.7`
- `numpy==1.26.4`, `scipy==1.11.4`, `pandas==2.2.1`, `pyarrow==14.0.2`
- `nibabel==5.2.1`
- `torch==2.4.0` (`+cpu` on Linux), `torchvision==0.19.0`
- `hd-bet==2.0.0`
- `pyyaml==6.0.1`, `tqdm==4.66.2`, `click==8.1.7`
- `pytest==8.1.1`, `pytest-cov==5.0.0`

### Dataset

The pipeline targets the **Kaggle WMH Segmentation Challenge** dataset
(Amsterdam / Singapore / Utrecht) with paired T1, FLAIR, and expert binary lesion
masks in NIfTI format. Place it at:

```
datasets/kaggle/wmh_data/
├── training/
│   ├── Amsterdam/GE3T/<subject_id>/{pre/T1.nii.gz, pre/FLAIR.nii.gz, wmh.nii.gz}
│   ├── Singapore/<subject_id>/...
│   └── Utrecht/<subject_id>/...
└── test/
    └── ...
```

A `generic` layout is also supported via `--layout generic` for arbitrary
subject directories.

---

## 2. Installation

The setup script installs Python 3.11, OpenJDK 17, creates a venv, installs all
pinned Python dependencies, installs the `wmh_spark` package in editable mode,
and verifies the HD-BET binary.

```bash
cd wmh-spark
./scripts/setup.sh            # full install
# or:
./scripts/setup.sh --verify   # install + run smoke test
./scripts/setup.sh --skip-system   # skip Python/Java install (already present)
```

Then activate the venv:

```bash
source wmh-spark/.venv/bin/activate
```

Quick verification:

```bash
hd-bet --help                                            # HD-BET CLI
python -m wmh_spark.preprocessing.skull_strip --help     # Skull-strip CLI
python wmh-spark/scripts/smoke_test_e2e.py               # synthetic end-to-end
```

---

## 3. Project Layout

```
wmh-spark/                                # repo root
├── datasets/                             # Kaggle WMH data lives here
│   └── kaggle/wmh_data/{training,test}/
├── wmh-spark/                            # Python project
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── configs/local.yaml                # Spark + path defaults
│   ├── scripts/
│   │   ├── setup.sh                      # one-shot installer
│   │   ├── run_pipeline.py               # ★ end-to-end driver
│   │   ├── make_manifest.py              # build subject manifest.parquet
│   │   ├── smoke_test_io.py              # I/O smoke test
│   │   ├── smoke_test_batch.py           # HD-BET batch smoke test
│   │   └── smoke_test_e2e.py             # synthetic end-to-end smoke test
│   ├── src/wmh_spark/
│   │   ├── preprocessing/                # HD-BET runner + DSC quality gate
│   │   ├── feature_extraction.py         # voxel DataFrame builder
│   │   ├── models/                       # Random Forest training + prediction
│   │   ├── postprocessing/               # connected-component filtering
│   │   ├── evaluation/                   # Dice + benchmark logging
│   │   ├── reporting.py                  # per-subject + aggregate reports
│   │   ├── dataset_layout.py             # site/scanner discovery
│   │   ├── io_utils.py
│   │   └── benchmark.py
│   ├── tests/{unit,integration}/
│   └── data/                             # default work / output root
```

---

## 4. End-to-End Pipeline

The driver is [wmh-spark/scripts/run_pipeline.py](wmh-spark/scripts/run_pipeline.py).
It runs all 10 stages in order:

1. Arrange dataset → per-split `manifest.parquet`
2. HD-BET skull-strip (train + test)
3. Build per-subject spatial prior from FLAIR brain mask
4. Build voxel feature DataFrame, union across train subjects
5. Train Spark MLlib Random Forest, persist model
6. (Optional) Tune RF probability threshold on a validation split
7. Predict + post-process per test subject → `wmh_pred.nii.gz`
8. Compute Dice + lesion stats per subject
9. Render per-subject overlay PNG + `report.html`
10. Write aggregate report, metrics CSV, baseline comparisons, benchmarks JSON

### Run a small pilot (5 train + 5 test subjects)

```bash
source wmh-spark/.venv/bin/activate

python wmh-spark/scripts/run_pipeline.py \
    --data-root datasets/kaggle/wmh_data \
    --layout kaggle-wmh \
    --output-root wmh-spark/data/output/runs/pilot \
    --train-sites Amsterdam,Singapore \
    --test-sites Amsterdam,Singapore \
    --scanners GE3T \
    --max-train 5 --max-test 5 \
    --device auto
```

### Run the full 10×10 distributed configuration

```bash
python wmh-spark/scripts/run_pipeline.py \
    --data-root datasets/kaggle/wmh_data \
    --output-root wmh-spark/data/output/runs/run_10x10 \
    --max-train 10 --max-test 10 \
    --feature-set rich \
    --num-trees 20 --max-depth 8 \
    --negative-sampling-ratio 5.0 \
    --validation-fraction 0.2 \
    --candidate-thresholds 0.15,0.25,0.35,0.5,0.65 \
    --min-cluster-size 10 \
    --pipeline-engine distributed_sparse \
    --subject-parallelism 4 \
    --spark-driver-memory 8g \
    --spark-shuffle-partitions 16 \
    --device auto
```

### Reusing skull-strip outputs

Skull stripping is the slowest stage. Cache and reuse it across runs:

```bash
# first run produces <output-root>/skull_stripped/
python wmh-spark/scripts/run_pipeline.py ... --output-root data/output/runs/v1

# later runs reuse the same volumes
python wmh-spark/scripts/run_pipeline.py ... \
    --output-root data/output/runs/v2 \
    --skull-strip-root data/output/runs/v1/skull_stripped \
    --skip-skull-strip
```

### Pre-flight only (validate dataset, no HD-BET)

```bash
python wmh-spark/scripts/run_pipeline.py ... --preflight-only --show-subjects
```

---

## 5. Key CLI Flags

Run `python wmh-spark/scripts/run_pipeline.py --help` for the full list. The most
important ones:

| Flag | Default | Purpose |
|------|---------|---------|
| `--data-root` *(required)* | — | Path to the Kaggle WMH dataset root |
| `--layout` | `kaggle-wmh` | `kaggle-wmh` or `generic` |
| `--output-root` *(required)* | — | Where the run writes everything |
| `--train-sites` / `--test-sites` | `Amsterdam,Singapore` | Sites to include |
| `--scanners` | `GE3T` | Scanner sub-dir filter (`all` to keep every scanner) |
| `--max-train` / `--max-test` | `5` | Cap subjects per split |
| `--device` | `auto` | HD-BET device: `auto` / `cpu` / `cuda` / `mps` |
| `--enable-tta` | off | HD-BET test-time augmentation (slower, GPU-only really) |
| `--feature-set` | `rich` | Feature bundle for RF: `baseline` or `rich` |
| `--num-trees` / `--max-depth` | `20` / `8` | Random Forest hyper-params |
| `--negative-sampling-ratio` | `5.0` | Negatives kept per positive during training |
| `--prediction-threshold` | `0.25` | Class-1 RF probability threshold |
| `--candidate-thresholds` | `0.15,0.25,0.35,0.5,0.65` | Validation-time threshold sweep |
| `--validation-fraction` | `0.2` | Fraction of train held out for threshold tuning |
| `--min-cluster-size` | `10` | Connected-component voxel-count floor |
| `--pipeline-engine` | `legacy` | `legacy` or `distributed_sparse` (sparse brain-only voxel tasks) |
| `--subject-parallelism` | `0` (auto) | Concurrent subject tasks (skull-strip + sparse engine) |
| `--spark-master` | `local[N]` | Override Spark master URL |
| `--spark-driver-memory` | `6g` | Spark driver heap |
| `--spark-shuffle-partitions` | `8` | `spark.sql.shuffle.partitions` |
| `--skip-skull-strip` / `--skull-strip-root` | — | Reuse cached HD-BET outputs |
| `--preflight-only` | off | Validate layout + affines/shapes, then exit |

---

## 6. Outputs

After a successful run the `--output-root` directory contains:

```
<output-root>/
├── manifests/{train,validation,test}_manifest.parquet
├── skull_stripped/<subject_id>/
│   ├── <T1>_bet.nii.gz            # skull-stripped T1
│   ├── <FLAIR>_bet.nii.gz         # skull-stripped FLAIR
│   ├── <FLAIR>_bet_mask.nii.gz    # binary brain mask
│   ├── spatial_prior.nii.gz       # per-subject lesion-frequency prior
│   └── wmh_binary.nii.gz          # binarised GT (label 2 → 0)
├── model/                         # Spark MLlib Random Forest
├── predictions/<subject_id>/wmh_pred.nii.gz
├── per_subject/<subject_id>/{overlay.png, report.html}
├── aggregate/{report.html, metrics.csv, plots/}
└── benchmarks/
    ├── subject_bench.jsonl        # per-subject elapsed seconds
    ├── baseline_summaries.json    # all-zero / all-positive / spatial-prior baselines
    └── summary.json               # full run summary (mean Dice, hyper-params, totals)
```

---

## 7. Running the Test Suite

```bash
source wmh-spark/.venv/bin/activate
cd wmh-spark

pytest tests/unit/                       # fast unit tests
pytest tests/integration/ -m integration # slower e2e (Spark + HD-BET)
pytest                                   # everything
```

Standalone smoke tests (no pytest required):

```bash
python scripts/smoke_test_io.py     # I/O + manifest
python scripts/smoke_test_batch.py  # HD-BET on the bundled mini-dataset
python scripts/smoke_test_e2e.py    # synthetic end-to-end
```

---

## 8. Troubleshooting

- **`PYSPARK_PYTHON` mismatch / `Python 3.x` worker errors** — driver and workers
  must use the same Python minor version. The driver script sets
  `PYSPARK_PYTHON` from `sys.executable`, so always launch it via
  `wmh-spark/.venv/bin/python`.
- **HD-BET says GPU not found with `--device cuda`** — the CPU torch wheel is
  installed by default on Linux. Re-install torch with a CUDA build, or use
  `--device cpu` / `--device auto`.
- **Spark OOM during shuffle** — bump `--spark-driver-memory`, raise
  `--spark-shuffle-partitions`, or lower `--max-train` / use the
  `distributed_sparse` engine (sparse brain-only voxel rows).
- **Java not found** — re-run `./scripts/setup.sh` (installs OpenJDK 17) or
  export `JAVA_HOME` to a JDK 17 install before launching.
- **Mean Dice low / all-zero predictions** — drop `--prediction-threshold`,
  increase `--validation-fraction` to enable threshold tuning, or switch
  `--feature-set rich`.

