"""Benchmark harness: strong + weak scaling and Spark vs serial baselines.

We run the same pipeline under controlled configurations and emit a JSON
record with timings. Plots are produced offline (notebooks/scaling_plots.py).

Strong scaling   : fixed cohort, vary worker count.
Weak scaling     : per-worker work fixed, vary worker count and cohort size
                   proportionally.
Serial baseline  : same Python pipeline with no Spark (multiprocessing.Pool=1).
Multiproc baseline: same Python pipeline with multiprocessing.Pool on 1 node.

Honest scaling reporting requires *all four* of these. Otherwise a Spark
"win" might just be Python beating MATLAB, not Spark beating multiprocessing.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRecord:
    label: str
    backend: str           # "spark" | "serial" | "multiprocessing"
    n_subjects: int
    n_workers: int
    cores_per_worker: int
    wall_seconds: float
    throughput_subjects_per_min: float
    notes: Optional[str] = None


def run_strong_scaling(
    cfg_template: Config,
    worker_counts: list[int],
    out_path: str,
) -> list[BenchmarkRecord]:
    """Run the full pipeline at a fixed cohort size with varying worker counts."""
    from dataclasses import replace
    from .pipeline import run_pipeline

    records: list[BenchmarkRecord] = []
    for n_workers in worker_counts:
        spark_cfg = replace(cfg_template.spark, num_executors=n_workers)
        cfg = replace(cfg_template, spark=spark_cfg)
        logger.info("Strong scaling: %d workers", n_workers)
        t0 = time.perf_counter()
        summary = run_pipeline(cfg)
        wall = time.perf_counter() - t0
        n_subj = summary["counts"]["subjects"]
        records.append(BenchmarkRecord(
            label=f"strong_{n_workers}",
            backend="spark",
            n_subjects=n_subj,
            n_workers=n_workers,
            cores_per_worker=cfg.spark.executor_cores,
            wall_seconds=wall,
            throughput_subjects_per_min=60 * n_subj / wall,
        ))
        _persist(records, out_path)
    return records


def run_weak_scaling(
    cfg_template: Config,
    worker_counts: list[int],
    subjects_per_worker: int,
    manifest_paths: dict[int, str],
    out_path: str,
) -> list[BenchmarkRecord]:
    """Vary cohort size proportionally with worker count.

    ``manifest_paths`` maps n_workers -> a Parquet manifest containing exactly
    ``n_workers * subjects_per_worker`` subjects. The caller must build these
    manifests in advance using scripts/make_manifest.py with a subject limit.
    """
    from dataclasses import replace
    from .pipeline import run_pipeline

    records: list[BenchmarkRecord] = []
    for n_workers in worker_counts:
        manifest = manifest_paths[n_workers]
        spark_cfg = replace(cfg_template.spark, num_executors=n_workers)
        paths_cfg = replace(cfg_template.paths, manifest_path=manifest)
        cfg = replace(cfg_template, spark=spark_cfg, paths=paths_cfg)
        logger.info("Weak scaling: %d workers, manifest=%s", n_workers, manifest)
        t0 = time.perf_counter()
        summary = run_pipeline(cfg)
        wall = time.perf_counter() - t0
        n_subj = summary["counts"]["subjects"]
        records.append(BenchmarkRecord(
            label=f"weak_{n_workers}",
            backend="spark",
            n_subjects=n_subj,
            n_workers=n_workers,
            cores_per_worker=cfg.spark.executor_cores,
            wall_seconds=wall,
            throughput_subjects_per_min=60 * n_subj / wall,
        ))
        _persist(records, out_path)
    return records


def _persist(records: list[BenchmarkRecord], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)
