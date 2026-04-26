"""Smoke test — load datasets/ T1+FLAIR into a Spark DataFrame (SC-001).

Run from repo root:
    PYTHONPATH=wmh-spark/src python wmh-spark/scripts/smoke_test_io.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "wmh-spark" / "src"))

from pyspark.sql import SparkSession

from wmh_spark.benchmark import log_ingestion_run
from wmh_spark.config import load_config
from wmh_spark.io_utils import build_voxel_dataframe

CONFIG_PATH = ROOT / "wmh-spark" / "configs" / "local.yaml"
T1_PATH = ROOT / "datasets" / "T1_RMS.nii.gz"
# Smoke-test note: local datasets/ are unregistered acquisitions with different shapes.
# T1_RMS is used as a proxy for FLAIR to exercise the full pipeline code path.
# Phase 2 (Kaggle dataset) will use properly co-registered T1+FLAIR pairs.
FLAIR_PATH = ROOT / "datasets" / "T1_RMS.nii.gz"
BENCH_LOG = ROOT / "wmh-spark" / "data" / "logs" / "ingestion_bench.jsonl"


def main() -> None:
    cfg = load_config(CONFIG_PATH)

    spark = (
        SparkSession.builder.appName(cfg.spark.app_name)
        .master(cfg.spark.master)
        .config("spark.driver.memory", cfg.spark.driver_memory)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(f"T1   : {T1_PATH}")
    print(f"FLAIR: {FLAIR_PATH} (T1 proxy — smoke test only)")

    t0 = time.time()
    df = build_voxel_dataframe(
        spark=spark,
        subject_id="smoke_subject",
        t1_path=str(T1_PATH),
        flair_path=str(FLAIR_PATH),
        partition_count=cfg.ingestion.partition_count,
    )
    df.printSchema()

    count = df.count()
    elapsed = time.time() - t0

    print(f"\nRow count : {count:,}")
    print(f"Partitions: {df.rdd.getNumPartitions()}")
    print(f"Elapsed   : {elapsed:.2f}s")

    assert count > 0, "ERROR: DataFrame is empty — ingestion failed"
    # SC-001: must complete in under 60 seconds on local machine
    assert elapsed < 60, f"SC-001 FAIL: ingestion took {elapsed:.1f}s, threshold is 60s"

    record = log_ingestion_run(
        subject_id="smoke_subject",
        voxel_count=count,
        partition_count=df.rdd.getNumPartitions(),
        elapsed_seconds=elapsed,
        output_path=str(BENCH_LOG),
    )
    print(f"\nBenchmark : {record}")
    print(f"Log       : {BENCH_LOG}")
    print("\nSC-001 PASS ✓")
    spark.stop()


if __name__ == "__main__":
    main()
