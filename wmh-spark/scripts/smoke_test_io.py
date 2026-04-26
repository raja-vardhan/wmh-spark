"""Smoke test — load datasets/ T1+FLAIR into a Spark DataFrame (SC-001).

Run from repo root:
    PYTHONPATH=wmh-spark/src python wmh-spark/scripts/smoke_test_io.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

# Pin PySpark workers to the same interpreter as the driver — otherwise
# Spark spawns workers with the system `python3` which may be a different
# minor version (e.g. 3.12 vs venv 3.11) and tasks fail with
# PYTHON_VERSION_MISMATCH.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# Force Java 17 — Spark 3.5 officially supports 8/11/17. Java 21 (system
# default on noble) sealed the sun.misc.Unsafe / DirectByteBuffer paths
# that Arrow 14 uses, breaking Arrow-based pandas→Spark conversion even
# with `--add-opens` flags.
_JAVA_17_CANDIDATES = [
    "/usr/lib/jvm/java-17-openjdk-amd64",
    *sorted(Path("/usr/lib/jvm").glob("java-17-openjdk*")),
    *sorted(Path("/usr/lib/jvm").glob("temurin-17-jdk-*")),
    *sorted(Path("/usr/lib/jvm").glob("zulu17-*")),
]
for _candidate in _JAVA_17_CANDIDATES:
    if Path(_candidate).is_dir():
        os.environ["JAVA_HOME"] = str(_candidate)
        os.environ["PATH"] = f"{_candidate}/bin:{os.environ.get('PATH', '')}"
        break

# Open module internals that Spark + Arrow need on Java 17. Propagated to
# every JVM PySpark forks (driver and executor share one JVM in local mode).
_JVM_OPENS = (
    "--add-opens=java.base/java.lang=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
    "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
    "--add-opens=java.base/java.io=ALL-UNNAMED "
    "--add-opens=java.base/java.net=ALL-UNNAMED "
    "--add-opens=java.base/java.nio=ALL-UNNAMED "
    "--add-opens=java.base/java.util=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
    "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
    "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
    "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
    "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED"
)
os.environ.setdefault("JAVA_TOOL_OPTIONS", _JVM_OPENS)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "wmh-spark" / "src"))

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from wmh_spark.benchmark import log_ingestion_run
from wmh_spark.config import load_config
from wmh_spark.io_utils import derive_partition_count, load_volume

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_test_io")

CONFIG_PATH = ROOT / "wmh-spark" / "configs" / "local.yaml"
T1_PATH = ROOT / "datasets" / "T1_RMS.nii.gz"
# Smoke-test note: local datasets/ are unregistered acquisitions with different shapes.
# T1_RMS is used as a proxy for FLAIR to exercise the full pipeline code path.
# Phase 2 (Kaggle dataset) will use properly co-registered T1+FLAIR pairs.
FLAIR_PATH = ROOT / "datasets" / "T1_RMS.nii.gz"
BENCH_LOG = ROOT / "wmh-spark" / "data" / "logs" / "ingestion_bench.jsonl"

STAGES = [
    "load config",
    "start Spark session",
    "load T1 volume",
    "load FLAIR volume",
    "validate paired volumes",
    "flatten + build pandas DF",
    "create Spark DataFrame",
    "count voxels (Spark action)",
    "write benchmark",
]


def main() -> None:
    t_total = time.time()
    with logging_redirect_tqdm():
        bar = tqdm(
            total=len(STAGES),
            desc="smoke",
            unit="stage",
            bar_format="{desc} |{bar:25}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        )

        # 1. config -------------------------------------------------------
        bar.set_postfix_str(STAGES[0])
        cfg = load_config(CONFIG_PATH)
        log.info(
            "config: master=%s driver_memory=%s partition_count=%s",
            cfg.spark.master, cfg.spark.driver_memory, cfg.ingestion.partition_count,
        )
        bar.update(1)

        # 2. spark --------------------------------------------------------
        bar.set_postfix_str(STAGES[1])
        t_spark = time.time()
        spark = (
            SparkSession.builder.appName(cfg.spark.app_name)
            .master(cfg.spark.master)
            .config("spark.driver.memory", cfg.spark.driver_memory)
            # Arrow makes pandas→Spark conversion ~50× faster on 22M rows
            # (333s → <10s). Without it, createDataFrame falls back to
            # row-by-row Python serialisation.
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
        log.info("Spark %s ready in %.2fs", spark.version, time.time() - t_spark)
        bar.update(1)

        t0 = time.time()

        # 3. load T1 ------------------------------------------------------
        bar.set_postfix_str(f"{STAGES[2]} ({T1_PATH.name})")
        t_t1 = time.time()
        t1, t1_aff = load_volume(T1_PATH)
        log.info(
            "T1 loaded: shape=%s dtype=%s size=%.1f MB in %.2fs",
            t1.shape, t1.dtype, t1.nbytes / 1e6, time.time() - t_t1,
        )
        bar.update(1)

        # 4. load FLAIR ---------------------------------------------------
        bar.set_postfix_str(f"{STAGES[3]} ({FLAIR_PATH.name})")
        t_fl = time.time()
        flair, flair_aff = load_volume(FLAIR_PATH)
        log.info(
            "FLAIR loaded: shape=%s dtype=%s size=%.1f MB in %.2fs",
            flair.shape, flair.dtype, flair.nbytes / 1e6, time.time() - t_fl,
        )
        bar.update(1)

        # 5. validate -----------------------------------------------------
        bar.set_postfix_str(STAGES[4])
        if t1.shape != flair.shape:
            raise ValueError(f"shape mismatch: T1 {t1.shape} != FLAIR {flair.shape}")
        if not np.allclose(t1_aff, flair_aff, atol=1e-4):
            raise ValueError("affine mismatch — volumes not co-registered")
        voxel_count = t1.size
        log.info("validation passed: %d voxels per volume", voxel_count)
        bar.update(1)

        # 6. flatten + pandas --------------------------------------------
        bar.set_postfix_str(f"{STAGES[5]} ({voxel_count:,} voxels)")
        t_flat = time.time()
        z_idx, y_idx, x_idx = np.indices(t1.shape, dtype=np.int32)
        pdf = pd.DataFrame({
            "subject_id": "smoke_subject",
            "x": x_idx.ravel(),
            "y": y_idx.ravel(),
            "z": z_idx.ravel(),
            "t1": t1.ravel(),
            "flair": flair.ravel(),
        })
        log.info(
            "pandas DF built: rows=%d mem=%.1f MB in %.2fs",
            len(pdf), pdf.memory_usage(deep=True).sum() / 1e6, time.time() - t_flat,
        )
        bar.update(1)

        # 7. spark createDataFrame ---------------------------------------
        n_parts = derive_partition_count(voxel_count, override=cfg.ingestion.partition_count)
        bar.set_postfix_str(f"{STAGES[6]} ({n_parts} partition{'s' if n_parts != 1 else ''})")
        t_sdf = time.time()
        df = spark.createDataFrame(pdf).repartition(n_parts)
        log.info("Spark DataFrame created in %.2fs (lazy, no action yet)", time.time() - t_sdf)
        df.printSchema()
        bar.update(1)

        # 8. count --------------------------------------------------------
        bar.set_postfix_str(STAGES[7])
        t_cnt = time.time()
        count = df.count()
        log.info("df.count() = %d in %.2fs", count, time.time() - t_cnt)
        bar.update(1)

        elapsed = time.time() - t0
        log.info("ingestion total: rows=%d partitions=%d elapsed=%.2fs",
                 count, df.rdd.getNumPartitions(), elapsed)

        assert count > 0, "ERROR: DataFrame is empty — ingestion failed"
        # SC-001: must complete in under 60 seconds on local machine
        assert elapsed < 60, f"SC-001 FAIL: ingestion took {elapsed:.1f}s, threshold is 60s"

        # 9. benchmark ----------------------------------------------------
        bar.set_postfix_str(STAGES[8])
        record = log_ingestion_run(
            subject_id="smoke_subject",
            voxel_count=count,
            partition_count=df.rdd.getNumPartitions(),
            elapsed_seconds=elapsed,
            output_path=str(BENCH_LOG),
        )
        log.info("benchmark logged → %s", BENCH_LOG)
        bar.update(1)
        bar.close()

        spark.stop()
        log.info("SC-001 PASS ✓ (total wall: %.2fs)", time.time() - t_total)


if __name__ == "__main__":
    main()
