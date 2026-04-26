"""Central run configuration.

All knobs live here so a single YAML maps to a frozen dataclass tree.
Log the config alongside every run output to guarantee reproducibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class SparkConfig:
    app_name: str = "wmh-spark"
    master: str = "local[4]"
    driver_memory: str = "8g"
    executor_memory: str = "12g"
    executor_cores: int = 4
    num_executors: int = 4
    extra_conf: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionConfig:
    # 0 means auto-derive from voxel count and 16 GB/worker ceiling (FR-006).
    partition_count: int = 0


@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "data"
    manifest_path: str = "data/manifest.parquet"
    work_dir: str = "data/work"
    output_dir: str = "data/output"
    model_dir: str = "data/output/models"
    metrics_dir: str = "data/output/metrics"
    log_dir: str = "data/logs"


@dataclass(frozen=True)
class Config:
    seed: int = 42
    spark: SparkConfig = field(default_factory=SparkConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _build(cls, data: Optional[dict]):
    if not data:
        return cls()
    valid = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in data.items() if k in valid})


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        seed=raw.get("seed", 42),
        spark=_build(SparkConfig, raw.get("spark")),
        ingestion=_build(IngestionConfig, raw.get("ingestion")),
        paths=_build(PathsConfig, raw.get("paths")),
    )


def dump_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
