# WMH-Spark: Scalable White Matter Hyperintensity Quantification on Apache Spark

A distributed, license-free re-implementation of the UBO Detector pipeline for
quantifying White Matter Hyperintensities (WMH) from FLAIR + T1 MRI, built on
Apache Spark and PySpark MLlib.

## Why this exists

The "gold standard" UBO Detector is a single-threaded MATLAB pipeline that does
not scale to modern population cohorts (UK Biobank, ADNI). This project
re-engineers the same algorithm as a Spark application that:

1. Parallelizes **across subjects** using an RDD-of-subjects pattern (the only
   form of parallelism that genuinely fits volumetric MRI workloads).
2. Trains a single global classifier (Random Forest, k-NN baseline, XGBoost)
   on subsampled voxels pooled across subjects via a Spark DataFrame, where
   distributed training actually pays off.
3. Broadcasts the trained model back to workers for per-subject inference.
4. Replaces MATLAB / proprietary dependencies with NumPy, nibabel, scikit-learn,
   and Spark MLlib.

## Architecture (subject-level parallelism, not voxel-level)

```
              ┌─────────────────────────────────────────────┐
              │       Driver  (orchestration only)          │
              └─────────────────────────────────────────────┘
                                  │
        ┌─────────────┬───────────┼───────────┬─────────────┐
        ▼             ▼           ▼           ▼             ▼
   ┌────────┐    ┌────────┐  ┌────────┐  ┌────────┐    ┌────────┐
   │Worker 1│    │Worker 2│  │Worker 3│  │Worker 4│    │  ...   │
   │subj A,E│    │subj B,F│  │subj C,G│  │subj D,H│    │        │
   └────────┘    └────────┘  └────────┘  └────────┘    └────────┘
        │             │           │           │             │
        └─────────────┴─────┬─────┴───────────┴─────────────┘
                            ▼
              ┌─────────────────────────────────────────────┐
              │  Pooled subsampled voxel DataFrame          │
              │  → MLlib RandomForestClassifier (.fit)      │
              └─────────────────────────────────────────────┘
                            │ broadcast model
                            ▼
              Per-subject inference on workers → masks → metrics
```

## Pipeline stages

Each stage is a Spark job that reads from and writes to a shared filesystem
(local FS for laptop runs, Lustre/NFS on CHPC, S3 for cloud).

| Stage | Input | Output | Spark role |
|------|-------|--------|------------|
| 01 ingest    | Kaggle / WMH Challenge raw NIfTI | manifest of subject paths | DataFrame |
| 02 strip     | T1 + FLAIR per subject | brain-extracted volumes | RDD map |
| 03 register  | stripped volumes        | MNI-aligned volumes + warp | RDD map |
| 04 features  | aligned volumes         | per-voxel feature ndarrays  | RDD map |
| 05 train     | pooled feature samples  | fitted model + metadata     | MLlib  |
| 06 infer     | features + model bcast  | predicted lesion masks      | RDD map |
| 07 evaluate  | masks + ground truth    | per-subject + cohort metrics| DataFrame |

## Key design decisions

- **Subject-as-row, not voxel-as-row.** Each RDD record is one subject; the
  worker loads the volume into NumPy and processes it locally. Voxel-level
  DataFrames blow up serialization cost without any shuffle benefit.
- **Single global classifier, not per-subject.** Training pools subsampled
  voxels across the entire cohort. This is where Spark earns its keep.
- **Class imbalance handled explicitly.** WMH voxels are <1% of brain volume.
  We subsample negatives to a configurable ratio (default 5:1) and report
  lesion-wise F1 in addition to voxel Dice.
- **Skull stripping is a separate pre-stage**, timed independently so
  Spark-vs-MATLAB benchmarks measure what they claim to measure.
- **Evaluation uses expert ground truth as primary**, UBO output as secondary.
  No circular validation.

## Repository layout

```
wmh-spark/
├── src/wmh_spark/
│   ├── config.py              # central run config (dataclass)
│   ├── io_utils.py            # NIfTI load/save, manifest IO
│   ├── stripping.py           # HD-BET / SynthStrip wrappers
│   ├── registration.py        # MNI registration via ANTs
│   ├── features.py            # per-voxel feature extraction
│   ├── sampling.py            # class-balanced voxel sampling
│   ├── models/
│   │   ├── knn_baseline.py    # UBO-style k-NN reproduction
│   │   ├── rf_spark.py        # Spark MLlib Random Forest
│   │   └── xgb_spark.py       # SparkXGBoost wrapper
│   ├── inference.py           # broadcast-model inference
│   ├── evaluation.py          # Dice, lesion F1, Hausdorff, vol diff
│   ├── pipeline.py            # end-to-end orchestration
│   └── benchmark.py           # timing + scaling harness
├── scripts/
│   ├── run_pipeline.py        # main entrypoint (spark-submit)
│   ├── run_benchmark.py       # strong + weak scaling harness
│   └── make_manifest.py       # build subject manifest from data dir
├── configs/
│   ├── local.yaml             # laptop / single-node debug
│   ├── chpc_4node.yaml        # CHPC SLURM 4-worker config
│   └── chpc_8node.yaml        # for weak-scaling experiments
├── tests/                     # pytest unit tests
└── docs/
    └── architecture.md
```

## Reproducibility

- All random seeds fixed in `config.py` and committed to version control.
- Dataset SHA-256 checksums in `data/checksums.txt`.
- Exact Java + Spark + Python versions pinned in `requirements.txt` and
  documented in `docs/environment.md` with the CHPC `module load` sequence.
- Train/val/test subject splits committed as `configs/splits/*.txt`.

## Quick start (local dev)

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Build subject manifest from a data directory
python scripts/make_manifest.py --data-dir ./data/raw --out ./data/manifest.parquet

# 3. Run end-to-end pipeline
spark-submit \
    --master local[4] \
    --conf spark.driver.memory=8g \
    scripts/run_pipeline.py --config configs/local.yaml
```

## CHPC (SLURM) submission

```bash
sbatch scripts/slurm/spark_4node.sbatch
```

See `docs/chpc_setup.md` for cluster bring-up details.
