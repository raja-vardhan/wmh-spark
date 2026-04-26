"""Build a Parquet subject manifest from a data directory.

Usage:
    python wmh-spark/scripts/make_manifest.py \\
        --data-dir datasets/ \\
        --output data/manifest.parquet
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from wmh_spark.io_utils import SubjectRecord


def find_subjects(data_dir: Path) -> list[SubjectRecord]:
    records = []
    for subj_dir in sorted(data_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        t1_files = sorted(glob.glob(str(subj_dir / "t1*.nii*")))
        flair_files = sorted(glob.glob(str(subj_dir / "flair*.nii*")))
        if not t1_files or not flair_files:
            continue
        mask_files = sorted(glob.glob(str(subj_dir / "wmh_mask*.nii*")))
        records.append(
            SubjectRecord(
                subject_id=subj_dir.name,
                t1_path=t1_files[0],
                flair_path=flair_files[0],
                gt_mask_path=mask_files[0] if mask_files else None,
            )
        )
    return records


def write_manifest(records: list[SubjectRecord], output: Path) -> None:
    rows = [r.to_dict() for r in records]
    schema = pa.schema(
        [
            ("subject_id", pa.string()),
            ("t1_path", pa.string()),
            ("flair_path", pa.string()),
            ("gt_mask_path", pa.string()),
            ("site", pa.string()),
            ("age", pa.float64()),
            ("split", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output))
    print(f"Wrote {len(records)} subjects to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build subject manifest Parquet file")
    parser.add_argument("--data-dir", required=True, help="Root directory of subject folders")
    parser.add_argument("--output", required=True, help="Output manifest.parquet path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"ERROR: data-dir not found: {data_dir}")

    records = find_subjects(data_dir)
    if not records:
        raise SystemExit(f"ERROR: no subjects found in {data_dir} (need t1*.nii* + flair*.nii*)")

    write_manifest(records, Path(args.output))


if __name__ == "__main__":
    main()
