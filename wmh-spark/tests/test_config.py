"""Tests for config.py — T008."""

from __future__ import annotations

import pytest
import yaml

from wmh_spark.config import Config, IngestionConfig, SparkConfig, dump_config, load_config


def test_load_config_roundtrip(tmp_path):
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "spark": {"master": "local[8]", "driver_memory": "4g"},
                "ingestion": {"partition_count": 12},
                "paths": {"data_dir": "/data/test"},
            }
        )
    )
    cfg = load_config(cfg_path)
    assert cfg.spark.master == "local[8]"
    assert cfg.spark.driver_memory == "4g"
    assert cfg.ingestion.partition_count == 12
    assert cfg.paths.data_dir == "/data/test"


def test_load_config_missing_keys_use_defaults(tmp_path):
    cfg_path = tmp_path / "minimal.yaml"
    cfg_path.write_text("{}\n")
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    assert cfg.spark.master == "local[4]"
    assert cfg.ingestion.partition_count == 0


def test_ingestion_config_partition_count_default():
    cfg = IngestionConfig()
    assert cfg.partition_count == 0, "Default must be 0 (auto-derive)"


def test_dump_config_creates_file(tmp_path):
    cfg = Config()
    out = tmp_path / "sub" / "config.yaml"
    dump_config(cfg, out)
    assert out.exists()
    reloaded = load_config(out)
    assert reloaded.seed == cfg.seed


def test_spark_config_immutable():
    cfg = SparkConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.master = "local[1]"  # type: ignore[misc]
