"""Tests for config loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from wmh_spark.config import Config, dump_config, load_config


def test_load_default_local_yaml(tmp_path):
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "local.yaml"
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    assert cfg.seed == 42
    assert cfg.spark.master.startswith("local")
    assert cfg.model.name in {"rf", "knn", "xgb"}


def test_partial_yaml_uses_defaults(tmp_path):
    minimal = {"seed": 7}
    path = tmp_path / "min.yaml"
    path.write_text(yaml.safe_dump(minimal))
    cfg = load_config(path)
    assert cfg.seed == 7
    # Sub-configs default to their dataclass defaults.
    assert cfg.features.neighborhood_radius == 1


def test_dump_round_trip(tmp_path):
    cfg = Config()
    out = tmp_path / "resolved.yaml"
    dump_config(cfg, out)
    loaded = load_config(out)
    assert loaded == cfg


def test_unknown_keys_ignored(tmp_path):
    """Forward compatibility: extra keys in YAML shouldn't break loading."""
    path = tmp_path / "extras.yaml"
    path.write_text(yaml.safe_dump({
        "seed": 1, "model": {"name": "rf", "future_field": "ignored"}
    }))
    cfg = load_config(path)
    assert cfg.model.name == "rf"
