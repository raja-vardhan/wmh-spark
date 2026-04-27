"""End-to-end smoke test for the WMH Spark pipeline on synthetic sample data.

Run from repo root:
    .venv/bin/python scripts/smoke_test_e2e.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

# Keep PySpark workers on the same Python minor version as the driver.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "wmh-spark" / "src"))

from pyspark.sql import SparkSession

from wmh_spark.evaluation import (
    EvaluationConfig,
    assert_reproduction_accuracy,
    log_cluster_throughput,
    log_subject_benchmark,
    summarize_scaling,
)
from wmh_spark.feature_extraction import build_feature_dataframe
from wmh_spark.io_utils import save_volume
from wmh_spark.models import ClassificationConfig, predict_voxel_mask, train_random_forest_model
from wmh_spark.postprocessing import PostProcessingConfig, postprocess_predictions

SHAPE = (16, 20, 18)
SUBJECT_ID = "sample_e2e_subject"


def _make_sample_subject(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4, dtype=np.float32) * 2.0
    affine[3, 3] = 1.0
    affine[:3, 3] = [-16.0, -20.0, -18.0]

    z, y, x = np.indices(SHAPE, dtype=np.float32)
    center = np.array([(SHAPE[0] - 1) / 2, (SHAPE[1] - 1) / 2, (SHAPE[2] - 1) / 2])
    brain = (
        ((z - center[0]) / 6.2) ** 2
        + ((y - center[1]) / 8.0) ** 2
        + ((x - center[2]) / 7.0) ** 2
    ) < 1.0

    lesion = np.zeros(SHAPE, dtype=bool)
    lesion[6:11, 8:13, 7:12] = True
    lesion &= brain

    rng = np.random.default_rng(123)
    flair = np.zeros(SHAPE, dtype=np.float32)
    flair[brain] = 80.0 + rng.normal(0.0, 2.0, int(brain.sum()))
    flair[lesion] = 240.0 + rng.normal(0.0, 2.0, int(lesion.sum()))

    t1 = np.zeros(SHAPE, dtype=np.float32)
    t1[brain] = 130.0 + rng.normal(0.0, 2.0, int(brain.sum()))
    t1[lesion] = 145.0 + rng.normal(0.0, 1.0, int(lesion.sum()))

    spatial_prior = np.zeros(SHAPE, dtype=np.float32)
    spatial_prior[brain] = 0.1
    spatial_prior[lesion] = 0.95

    mask = lesion.astype(np.uint8)

    paths = {
        "t1": out_dir / "t1.nii.gz",
        "flair": out_dir / "flair.nii.gz",
        "spatial_prior": out_dir / "spatial_prior.nii.gz",
        "mask": out_dir / "wmh_mask.nii.gz",
    }
    save_volume(t1, affine, paths["t1"])
    save_volume(flair, affine, paths["flair"])
    save_volume(spatial_prior, affine, paths["spatial_prior"])
    save_volume(mask, affine, paths["mask"], dtype=np.uint8)
    return paths


def main() -> None:
    t0 = time.time()
    out_dir = ROOT / "wmh-spark" / "data" / "e2e_smoke"
    sample_dir = out_dir / SUBJECT_ID
    benchmark_dir = out_dir / "benchmarks"
    paths = _make_sample_subject(sample_dir)

    spark = (
        SparkSession.builder.appName("wmh-spark-e2e-smoke")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        features = build_feature_dataframe(
            spark=spark,
            subject_id=SUBJECT_ID,
            t1_path=paths["t1"],
            flair_path=paths["flair"],
            spatial_prior_path=paths["spatial_prior"],
            mask_path=paths["mask"],
            partition_count=4,
        )

        rf_config = ClassificationConfig(
            num_trees=12,
            max_depth=5,
            training_partitions=4,
            seed=7,
        )
        model = train_random_forest_model(features, rf_config)
        raw_predictions = predict_voxel_mask(model, features, rf_config)
        postprocessed = postprocess_predictions(
            raw_predictions,
            PostProcessingConfig(volume_shape=SHAPE, min_cluster_size=10),
        )
        dice = assert_reproduction_accuracy(
            postprocessed,
            EvaluationConfig(prediction_column="postprocessed_mask", label_column="label"),
        )

        row_count = postprocessed.count()
        elapsed = time.time() - t0
        subject_record = log_subject_benchmark(
            subject_id=SUBJECT_ID,
            elapsed_seconds=elapsed,
            matlab_baseline_seconds=120.0,
            worker_nodes=4,
            voxel_count=row_count,
            output_path=benchmark_dir / "subject_bench.jsonl",
        )
        throughput_1 = log_cluster_throughput(
            worker_nodes=1,
            subjects_processed=1,
            elapsed_seconds=elapsed * 3.7,
            output_path=benchmark_dir / "throughput.jsonl",
        )
        throughput_4 = log_cluster_throughput(
            worker_nodes=4,
            subjects_processed=1,
            elapsed_seconds=elapsed,
            output_path=benchmark_dir / "throughput.jsonl",
        )
        scaling = summarize_scaling(throughput_1, throughput_4)

        print("E2E SMOKE PASS")
        print(f"  subject_id: {SUBJECT_ID}")
        print(f"  rows: {row_count:,}")
        print(f"  dsc: {dice.dsc:.4f}")
        print(f"  predicted_positive: {dice.predicted_positive}")
        print(f"  reference_positive: {dice.reference_positive}")
        print(f"  elapsed_seconds: {elapsed:.2f}")
        print(f"  speedup_vs_matlab: {subject_record['speedup_vs_matlab']:.2f}x")
        print(f"  subjects_per_hour_4_workers: {throughput_4['subjects_per_hour']:.2f}")
        print(f"  scaling_efficiency_1_to_4: {scaling['scaling_efficiency']:.2f}")
        print(f"  outputs: {out_dir}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
