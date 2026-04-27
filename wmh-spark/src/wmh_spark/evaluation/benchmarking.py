"""Benchmark logging for end-to-end WMH pipeline evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_jsonl(record: dict, output_path: str | Path) -> dict:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


@dataclass(frozen=True)
class SubjectBenchmarkRecord:
    subject_id: str
    elapsed_seconds: float
    matlab_baseline_seconds: float
    speedup_vs_matlab: float
    reduced_latency: bool
    worker_nodes: int
    timestamp: str
    voxel_count: Optional[int] = None


@dataclass(frozen=True)
class ThroughputRecord:
    worker_nodes: int
    subjects_processed: int
    elapsed_seconds: float
    subjects_per_hour: float
    timestamp: str


def log_subject_benchmark(
    subject_id: str,
    elapsed_seconds: float,
    matlab_baseline_seconds: float,
    output_path: str | Path,
    worker_nodes: int = 1,
    voxel_count: Optional[int] = None,
) -> dict:
    """Append one per-subject end-to-end benchmark record as JSON Lines."""
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")
    if matlab_baseline_seconds <= 0:
        raise ValueError("matlab_baseline_seconds must be positive")
    if worker_nodes <= 0:
        raise ValueError("worker_nodes must be positive")

    record = SubjectBenchmarkRecord(
        subject_id=subject_id,
        elapsed_seconds=round(float(elapsed_seconds), 3),
        matlab_baseline_seconds=round(float(matlab_baseline_seconds), 3),
        speedup_vs_matlab=float(matlab_baseline_seconds) / float(elapsed_seconds),
        reduced_latency=bool(elapsed_seconds < matlab_baseline_seconds),
        worker_nodes=int(worker_nodes),
        voxel_count=voxel_count,
        timestamp=_timestamp(),
    )
    return _append_jsonl(asdict(record), output_path)


def log_cluster_throughput(
    worker_nodes: int,
    subjects_processed: int,
    elapsed_seconds: float,
    output_path: str | Path,
) -> dict:
    """Append one cluster throughput benchmark record as JSON Lines."""
    if worker_nodes <= 0:
        raise ValueError("worker_nodes must be positive")
    if subjects_processed <= 0:
        raise ValueError("subjects_processed must be positive")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be positive")

    subjects_per_hour = float(subjects_processed) / (float(elapsed_seconds) / 3600.0)
    record = ThroughputRecord(
        worker_nodes=int(worker_nodes),
        subjects_processed=int(subjects_processed),
        elapsed_seconds=round(float(elapsed_seconds), 3),
        subjects_per_hour=subjects_per_hour,
        timestamp=_timestamp(),
    )
    return _append_jsonl(asdict(record), output_path)


def summarize_scaling(
    baseline_record: dict,
    target_record: dict,
    near_linear_threshold: float = 0.75,
) -> dict:
    """Summarize throughput scaling from a baseline worker count to a target count."""
    if near_linear_threshold <= 0:
        raise ValueError("near_linear_threshold must be positive")

    baseline_workers = int(baseline_record["worker_nodes"])
    target_workers = int(target_record["worker_nodes"])
    baseline_throughput = float(baseline_record["subjects_per_hour"])
    target_throughput = float(target_record["subjects_per_hour"])

    if baseline_workers <= 0 or target_workers <= 0:
        raise ValueError("worker_nodes must be positive")
    if baseline_throughput <= 0 or target_throughput <= 0:
        raise ValueError("subjects_per_hour must be positive")
    if target_workers <= baseline_workers:
        raise ValueError("target worker count must exceed baseline worker count")

    throughput_speedup = target_throughput / baseline_throughput
    ideal_speedup = target_workers / baseline_workers
    scaling_efficiency = throughput_speedup / ideal_speedup
    return {
        "baseline_worker_nodes": baseline_workers,
        "target_worker_nodes": target_workers,
        "baseline_subjects_per_hour": baseline_throughput,
        "target_subjects_per_hour": target_throughput,
        "throughput_speedup": throughput_speedup,
        "ideal_speedup": ideal_speedup,
        "scaling_efficiency": scaling_efficiency,
        "near_linear_threshold": float(near_linear_threshold),
        "near_linear": bool(scaling_efficiency >= near_linear_threshold),
        "timestamp": _timestamp(),
    }
