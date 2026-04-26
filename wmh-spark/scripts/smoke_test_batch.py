"""Batch smoke test — process 10 synthetic subjects to verify SC-005 (no OOM).

Run from repo root:
    PYTHONPATH=wmh-spark/src python wmh-spark/scripts/smoke_test_batch.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "wmh-spark" / "src"))

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import SparkSession

from wmh_spark.io_utils import SubjectRecord, ingest_from_manifest, save_volume

AFFINE = np.eye(4, dtype=np.float32) * 2.0
AFFINE[3, 3] = 1.0
SHAPE = (24, 32, 28)


def _write_subject(base: Path, idx: int) -> SubjectRecord:
    subj = base / f"subj_{idx:03d}"
    subj.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(idx)
    save_volume(rng.random(SHAPE, dtype=np.float32), AFFINE, subj / "t1.nii.gz")
    save_volume(rng.random(SHAPE, dtype=np.float32), AFFINE, subj / "flair.nii.gz")
    return SubjectRecord(
        subject_id=f"subj_{idx:03d}",
        t1_path=str(subj / "t1.nii.gz"),
        flair_path=str(subj / "flair.nii.gz"),
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        records = [_write_subject(base, i) for i in range(10)]

        manifest_path = base / "manifest.parquet"
        rows = [r.to_dict() for r in records]
        pq.write_table(pa.Table.from_pylist(rows), str(manifest_path))

        spark = (
            SparkSession.builder.appName("wmh-spark-batch-smoke")
            .master("local[2]")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        results = ingest_from_manifest(spark, manifest_path, partition_count=2)

        assert len(results) == 10, f"Expected 10 subjects, got {len(results)}"
        for subject_id, df in results:
            cnt = df.count()
            assert cnt > 0, f"Subject {subject_id} produced empty DataFrame"
            print(f"  {subject_id}: {cnt:,} voxels")

        print(f"\nSC-005 PASS ✓  ({len(results)} subjects, no OOM)")
        spark.stop()


if __name__ == "__main__":
    main()
