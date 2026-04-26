"""Distributed voxel feature extraction for WMH classification."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from pyspark.sql import functions as F

from wmh_spark.io_utils import build_voxel_dataframe, load_volume, validate_subject_volumes

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_BYTES_PER_ROW = 48
DEFAULT_TARGET_PARTITION_BYTES = 256 * 1024**2
DEFAULT_MAX_PARTITION_BYTES = 16 * 1024**3
DEFAULT_MIN_FEATURE_PARTITIONS = 4


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
    t0 = time.time()
    t1, _, subject_affine = validate_subject_volumes(t1_path, flair_path)
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

    voxel_df = build_voxel_dataframe(
        spark=spark,
        subject_id=subject_id,
        t1_path=t1_path,
        flair_path=flair_path,
        partition_count=plan.partition_count,
        mask_path=mask_path,
    )
    ratio_df = add_t1_flair_ratio(voxel_df, ratio_epsilon=ratio_epsilon)
    prior_df = _spatial_prior_array_to_dataframe(
        spark=spark,
        spatial_prior=spatial_prior,
        partition_count=plan.partition_count,
    )
    joined = ratio_df.join(F.broadcast(prior_df), on=["x", "y", "z"], how="left")

    ordered_columns = [
        "subject_id",
        "x",
        "y",
        "z",
        "t1",
        "flair",
        "t1_flair_ratio",
        "spatial_prior",
    ]
    if "label" in joined.columns:
        ordered_columns.append("label")

    result = joined.select(*ordered_columns)
    logger.info(
        "build_feature_dataframe: subject=%s voxels=%d partitions=%d elapsed=%.2fs",
        subject_id,
        t1.size,
        plan.partition_count,
        time.time() - t0,
    )
    return result
