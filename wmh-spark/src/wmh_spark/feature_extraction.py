"""Distributed voxel feature extraction for WMH classification."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from scipy import ndimage
from pyspark.sql import functions as F

from wmh_spark.io_utils import load_volume, validate_binary_mask, validate_subject_volumes

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_BYTES_PER_ROW = 48
DEFAULT_TARGET_PARTITION_BYTES = 256 * 1024**2
DEFAULT_MAX_PARTITION_BYTES = 16 * 1024**3
DEFAULT_MIN_FEATURE_PARTITIONS = 4
DEFAULT_LOCAL_WINDOW_SIZE = 3

BASELINE_FEATURE_COLUMNS = ("t1", "flair", "t1_flair_ratio", "spatial_prior")
RICH_FEATURE_COLUMNS = (
    *BASELINE_FEATURE_COLUMNS,
    "t1_zscore",
    "flair_zscore",
    "t1_local_mean",
    "t1_local_std",
    "flair_local_mean",
    "flair_local_std",
    "x_norm",
    "y_norm",
    "z_norm",
    "distance_to_center",
)
FEATURE_SETS = {
    "baseline": BASELINE_FEATURE_COLUMNS,
    "rich": RICH_FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class FeaturePartitionPlan:
    """Resolved partitioning strategy for feature-generation DataFrames."""

    voxel_count: int
    partition_count: int
    bytes_per_row: int
    total_estimated_bytes: int
    estimated_bytes_per_partition: int
    target_partition_bytes: int
    max_partition_bytes: int
    override: int = 0


def derive_feature_partition_count(
    voxel_count: int,
    bytes_per_row: int = DEFAULT_FEATURE_BYTES_PER_ROW,
    target_partition_bytes: int = DEFAULT_TARGET_PARTITION_BYTES,
    max_partition_bytes: int = DEFAULT_MAX_PARTITION_BYTES,
    min_partitions: int = DEFAULT_MIN_FEATURE_PARTITIONS,
    override: int = 0,
) -> FeaturePartitionPlan:
    """Resolve a conservative feature partition plan for Spark shuffles.

    The target partition size is intentionally much lower than the 16 GiB
    worker ceiling so shuffle spills have room for Spark's in-memory overhead.
    """
    if voxel_count < 0:
        raise ValueError("voxel_count must be non-negative")
    if bytes_per_row <= 0:
        raise ValueError("bytes_per_row must be positive")
    if target_partition_bytes <= 0:
        raise ValueError("target_partition_bytes must be positive")
    if max_partition_bytes <= 0:
        raise ValueError("max_partition_bytes must be positive")
    if min_partitions <= 0:
        raise ValueError("min_partitions must be positive")

    total_bytes = voxel_count * bytes_per_row
    if override > 0:
        partition_count = override
    else:
        by_target = max(1, math.ceil(total_bytes / target_partition_bytes))
        by_ceiling = max(1, math.ceil(total_bytes / max_partition_bytes))
        partition_count = max(min_partitions, by_target, by_ceiling)

    estimated_partition_bytes = (
        math.ceil(total_bytes / partition_count) if partition_count else total_bytes
    )
    if estimated_partition_bytes > max_partition_bytes:
        logger.warning(
            "feature partition override may exceed worker memory: estimated=%d max=%d",
            estimated_partition_bytes,
            max_partition_bytes,
        )

    plan = FeaturePartitionPlan(
        voxel_count=voxel_count,
        partition_count=partition_count,
        bytes_per_row=bytes_per_row,
        total_estimated_bytes=total_bytes,
        estimated_bytes_per_partition=estimated_partition_bytes,
        target_partition_bytes=target_partition_bytes,
        max_partition_bytes=max_partition_bytes,
        override=override,
    )
    logger.info(
        "feature partition plan: voxels=%d partitions=%d est_partition_bytes=%d",
        plan.voxel_count,
        plan.partition_count,
        plan.estimated_bytes_per_partition,
    )
    return plan


def validate_spatial_prior_template(
    subject_shape: Sequence[int],
    subject_affine: np.ndarray,
    spatial_prior_path: str | Path,
) -> np.ndarray:
    """Load and validate a spatial-prior template on the subject voxel grid."""
    prior, prior_affine = load_volume(spatial_prior_path)
    subject_shape = tuple(subject_shape)

    if prior.shape != subject_shape:
        raise ValueError(
            f"spatial prior shape mismatch: prior shape {prior.shape} != "
            f"subject shape {subject_shape}"
        )
    if not np.allclose(prior_affine, subject_affine, atol=1e-4):
        raise ValueError(
            "spatial prior affine mismatch: prior template must be on the same "
            "standard grid as the subject volumes"
        )
    return prior


def _spatial_prior_array_to_dataframe(
    spark: "SparkSession",
    spatial_prior: np.ndarray,
    partition_count: int,
) -> "DataFrame":
    import pandas as pd

    z_idx, y_idx, x_idx = np.indices(spatial_prior.shape, dtype=np.int32)
    pdf = pd.DataFrame(
        {
            "x": x_idx.ravel(),
            "y": y_idx.ravel(),
            "z": z_idx.ravel(),
            "spatial_prior": spatial_prior.ravel().astype(np.float32),
        }
    )
    return (
        spark.createDataFrame(pdf)
        # Repartition to the feature plan count so the broadcast side is built
        # in bounded chunks under the 16 GiB worker memory ceiling.
        .repartition(partition_count)
    )


def build_spatial_prior_dataframe(
    spark: "SparkSession",
    spatial_prior_path: str | Path,
    subject_shape: Sequence[int],
    subject_affine: np.ndarray,
    partition_count: int = 0,
) -> "DataFrame":
    """Build a coordinate DataFrame for the standard spatial-prior template."""
    prior = validate_spatial_prior_template(
        subject_shape=subject_shape,
        subject_affine=subject_affine,
        spatial_prior_path=spatial_prior_path,
    )
    resolved_partitions = (
        partition_count
        if partition_count > 0
        else derive_feature_partition_count(prior.size).partition_count
    )
    return _spatial_prior_array_to_dataframe(spark, prior, resolved_partitions)


def add_t1_flair_ratio(
    df: "DataFrame",
    ratio_epsilon: float = 1e-6,
) -> "DataFrame":
    """Append a Spark-vectorized T1/FLAIR ratio column."""
    if ratio_epsilon < 0:
        raise ValueError("ratio_epsilon must be non-negative")

    flair = F.col("flair").cast("double")
    t1 = F.col("t1").cast("double")
    return df.withColumn(
        "t1_flair_ratio",
        F.when(F.abs(flair) <= F.lit(float(ratio_epsilon)), F.lit(0.0)).otherwise(
            t1 / flair
        ),
    )


def resolve_feature_columns(feature_set: str) -> tuple[str, ...]:
    """Resolve a named feature set to the concrete ordered column tuple."""
    try:
        return FEATURE_SETS[feature_set]
    except KeyError as exc:
        known = ", ".join(sorted(FEATURE_SETS))
        raise ValueError(f"unknown feature_set={feature_set!r}; expected one of: {known}") from exc


def _brain_mask(t1: np.ndarray, flair: np.ndarray) -> np.ndarray:
    return ((np.abs(t1) > 1e-6) | (np.abs(flair) > 1e-6)).astype(np.uint8)


def _safe_zscore(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = volume[mask > 0]
    if masked.size == 0:
        return np.zeros(volume.shape, dtype=np.float32)
    mean = float(masked.mean())
    std = float(masked.std())
    if std <= 1e-6:
        return np.zeros(volume.shape, dtype=np.float32)
    standardized = (volume - mean) / std
    standardized = standardized.astype(np.float32, copy=False)
    standardized[mask == 0] = 0.0
    return standardized


def _local_mean_std(
    volume: np.ndarray,
    mask: np.ndarray,
    window_size: int = DEFAULT_LOCAL_WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")

    kernel = np.ones((window_size, window_size, window_size), dtype=np.float32)
    masked_volume = volume * mask
    voxel_counts = ndimage.convolve(mask.astype(np.float32), kernel, mode="constant", cval=0.0)
    voxel_sums = ndimage.convolve(masked_volume.astype(np.float32), kernel, mode="constant", cval=0.0)
    voxel_sq_sums = ndimage.convolve(
        np.square(masked_volume, dtype=np.float32),
        kernel,
        mode="constant",
        cval=0.0,
    )

    mean = np.zeros(volume.shape, dtype=np.float32)
    std = np.zeros(volume.shape, dtype=np.float32)
    valid = voxel_counts > 0
    mean[valid] = voxel_sums[valid] / voxel_counts[valid]
    variance = np.zeros(volume.shape, dtype=np.float32)
    variance[valid] = (voxel_sq_sums[valid] / voxel_counts[valid]) - np.square(mean[valid])
    variance = np.clip(variance, a_min=0.0, a_max=None)
    std[valid] = np.sqrt(variance[valid], dtype=np.float32)
    mean[mask == 0] = 0.0
    std[mask == 0] = 0.0
    return mean, std


def _normalized_coordinate_grids(shape: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_idx, y_idx, x_idx = np.indices(shape, dtype=np.float32)
    z_den = max(shape[0] - 1, 1)
    y_den = max(shape[1] - 1, 1)
    x_den = max(shape[2] - 1, 1)
    return (
        z_idx / float(z_den),
        y_idx / float(y_den),
        x_idx / float(x_den),
    )


def _distance_to_center(z_norm: np.ndarray, y_norm: np.ndarray, x_norm: np.ndarray) -> np.ndarray:
    distance = np.sqrt(
        np.square(z_norm - 0.5) + np.square(y_norm - 0.5) + np.square(x_norm - 0.5)
    )
    max_distance = float(np.sqrt(3.0) / 2.0)
    return (distance / max_distance).astype(np.float32)


def build_feature_dataframe(
    spark: "SparkSession",
    subject_id: str,
    t1_path: str | Path,
    flair_path: str | Path,
    spatial_prior_path: str | Path,
    partition_count: int = 0,
    mask_path: Optional[str | Path] = None,
    ratio_epsilon: float = 1e-6,
    min_partitions: int = DEFAULT_MIN_FEATURE_PARTITIONS,
) -> "DataFrame":
    """Build a distributed voxel feature matrix for one subject.

    Output columns:
        subject_id, x, y, z, t1, flair, t1_flair_ratio, spatial_prior [, label]
    """
    import pandas as pd

    t0 = time.time()
    t1, flair, subject_affine = validate_subject_volumes(t1_path, flair_path)
    spatial_prior = validate_spatial_prior_template(
        subject_shape=t1.shape,
        subject_affine=subject_affine,
        spatial_prior_path=spatial_prior_path,
    )
    plan = derive_feature_partition_count(
        voxel_count=t1.size,
        min_partitions=min_partitions,
        override=partition_count,
    )
    spark.conf.set("spark.sql.shuffle.partitions", str(plan.partition_count))

    label_col: Optional[np.ndarray] = None
    if mask_path is not None:
        label_col = validate_binary_mask(mask_path).ravel().astype(np.int32)

    mask = _brain_mask(t1, flair)
    t1_zscore = _safe_zscore(t1, mask)
    flair_zscore = _safe_zscore(flair, mask)
    t1_local_mean, t1_local_std = _local_mean_std(t1, mask)
    flair_local_mean, flair_local_std = _local_mean_std(flair, mask)
    z_norm, y_norm, x_norm = _normalized_coordinate_grids(t1.shape)
    center_distance = _distance_to_center(z_norm, y_norm, x_norm)
    ratio = np.zeros(t1.shape, dtype=np.float32)
    valid_ratio = np.abs(flair) > float(ratio_epsilon)
    ratio[valid_ratio] = (t1[valid_ratio] / flair[valid_ratio]).astype(np.float32, copy=False)

    z_idx, y_idx, x_idx = np.indices(t1.shape, dtype=np.int32)
    pdf = pd.DataFrame(
        {
            "subject_id": subject_id,
            "x": x_idx.ravel(),
            "y": y_idx.ravel(),
            "z": z_idx.ravel(),
            "t1": t1.ravel(),
            "flair": flair.ravel(),
            "t1_flair_ratio": ratio.ravel(),
            "spatial_prior": spatial_prior.ravel().astype(np.float32),
            "t1_zscore": t1_zscore.ravel(),
            "flair_zscore": flair_zscore.ravel(),
            "t1_local_mean": t1_local_mean.ravel(),
            "t1_local_std": t1_local_std.ravel(),
            "flair_local_mean": flair_local_mean.ravel(),
            "flair_local_std": flair_local_std.ravel(),
            "x_norm": x_norm.ravel(),
            "y_norm": y_norm.ravel(),
            "z_norm": z_norm.ravel(),
            "distance_to_center": center_distance.ravel(),
        }
    )
    if label_col is not None:
        pdf["label"] = label_col

    result = spark.createDataFrame(pdf).repartition(plan.partition_count)
    logger.info(
        "build_feature_dataframe: subject=%s voxels=%d partitions=%d elapsed=%.2fs",
        subject_id,
        t1.size,
        plan.partition_count,
        time.time() - t0,
    )
    return result
