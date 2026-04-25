"""Central run configuration.

All knobs live here so a single YAML maps cleanly to a frozen dataclass.
This keeps experiments reproducible: log the config, fix the seeds, ship it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional
import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparkConfig:
    """Spark session parameters. These are passed through to SparkConf."""

    app_name: str = "wmh-spark"
    master: str = "local[4]"
    driver_memory: str = "8g"
    executor_memory: str = "12g"
    executor_cores: int = 4
    num_executors: int = 4
    # Larger Arrow batches help when shipping NumPy arrays through Pandas UDFs.
    arrow_max_records_per_batch: int = 1_000
    # Required for nibabel/NumPy on workers.
    python_exec: str = "python3"
    extra_conf: dict = field(default_factory=dict)


@dataclass(frozen=True)
class StrippingConfig:
    """Skull-stripping stage. HD-BET on GPU, SynthStrip as CPU fallback."""

    method: Literal["hdbet", "synthstrip", "precomputed"] = "synthstrip"
    container_path: Optional[str] = None  # path to .sif Singularity image
    gpu: bool = False
    # If precomputed, expect <subject>_brain.nii.gz already on disk.


@dataclass(frozen=True)
class RegistrationConfig:
    """Linear MNI152 registration via ANTs (antspyx)."""

    template_path: str = "data/templates/MNI152_T1_1mm_brain.nii.gz"
    transform_type: Literal["Rigid", "Affine", "SyN"] = "Affine"
    cache_warps: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    """Per-voxel feature extraction. Mirrors UBO Detector's feature space."""

    use_flair_zscore: bool = True
    use_t1_intensity: bool = True
    use_flair_t1_ratio: bool = True
    use_distance_to_ventricle: bool = True
    use_mni_coords: bool = True
    use_neighborhood_stats: bool = True
    neighborhood_radius: int = 1  # 3x3x3 cube
    # Restrict candidate voxels to white matter to mirror UBO.
    wm_mask_threshold: float = 0.5
    flair_min_threshold: float = 0.7  # fraction of NAWM mean


@dataclass(frozen=True)
class SamplingConfig:
    """Class-imbalance handling for training set construction."""

    negative_to_positive_ratio: int = 5
    # Cap how many voxels per subject contribute to global training set.
    max_voxels_per_subject: int = 100_000
    # How many subjects to use for training vs. eval (rest is val).
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    """Classifier hyperparameters. We benchmark all three on identical features."""

    name: Literal["knn", "rf", "xgb"] = "rf"
    # k-NN baseline (mirrors UBO Detector exactly).
    knn_k: int = 5
    # Random Forest (Spark MLlib).
    rf_num_trees: int = 100
    rf_max_depth: int = 12
    rf_min_instances_per_node: int = 50
    rf_subsampling_rate: float = 0.7
    # XGBoost.
    xgb_max_depth: int = 8
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 200


@dataclass(frozen=True)
class EvaluationConfig:
    """Metrics + thresholds. We deliberately do *not* threshold against UBO."""

    primary_reference: Literal["expert", "ubo"] = "expert"
    decision_threshold: float = 0.5
    compute_dice: bool = True
    compute_lesion_f1: bool = True
    compute_hausdorff: bool = True
    compute_volume_difference: bool = True
    # Match WMH Challenge protocol.
    lesion_overlap_threshold: float = 0.0  # any voxel overlap counts as TP


@dataclass(frozen=True)
class PathsConfig:
    """All filesystem paths in one place. Override per environment."""

    data_dir: str = "data"
    manifest_path: str = "data/manifest.parquet"
    work_dir: str = "data/work"
    output_dir: str = "data/output"
    model_dir: str = "data/output/models"
    metrics_dir: str = "data/output/metrics"
    log_dir: str = "data/logs"


@dataclass(frozen=True)
class Config:
    """Top-level config aggregating all stages."""

    seed: int = 42
    spark: SparkConfig = field(default_factory=SparkConfig)
    stripping: StrippingConfig = field(default_factory=StrippingConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _build_dataclass(cls, data: dict):
    """Instantiate a dataclass from a (possibly partial) dict."""
    if data is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a YAML config file into a frozen Config dataclass."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        seed=raw.get("seed", 42),
        spark=_build_dataclass(SparkConfig, raw.get("spark")),
        stripping=_build_dataclass(StrippingConfig, raw.get("stripping")),
        registration=_build_dataclass(RegistrationConfig, raw.get("registration")),
        features=_build_dataclass(FeatureConfig, raw.get("features")),
        sampling=_build_dataclass(SamplingConfig, raw.get("sampling")),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        evaluation=_build_dataclass(EvaluationConfig, raw.get("evaluation")),
        paths=_build_dataclass(PathsConfig, raw.get("paths")),
    )


def dump_config(cfg: Config, path: str | Path) -> None:
    """Persist the resolved config alongside the run output for reproducibility."""
    with open(path, "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
