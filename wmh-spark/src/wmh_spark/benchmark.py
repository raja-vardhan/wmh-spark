"""Benchmark logging for pipeline runs (Constitution Principle V).

Phase 1: logs per-subject elapsed_seconds and partition_count.
Phase 2 TODO: extend with subjects_per_hour and 1→4 node scaling factor.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def log_ingestion_run(
    subject_id: str,
    voxel_count: int,
    partition_count: int,
    elapsed_seconds: float,
    output_path: str | Path,
) -> dict:
    """Append one JSON-lines record to the benchmark log and return the record."""
    record = {
        "subject_id": subject_id,
        "voxel_count": voxel_count,
        "partition_count": partition_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
