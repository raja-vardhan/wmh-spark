"""Distributed voxel feature extraction for WMH classification."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, Sequence

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


@dataclass(frozen=True)
class SubjectFeatureTask:
    """One subject's paths for distributed feature extraction."""

    subject_id: str
    t1_path: str
    flair_path: str
    spatial_prior_path: str
    mask_path: Optional[str] = None


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


def feature_row_schema(
    include_label: bool = False,
    include_shape_columns: bool = False,
):
    """Return the Spark schema for dense or sparse feature rows."""
    from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType

    fields = [
        StructField("subject_id", StringType(), nullable=False),
        StructField("x", IntegerType(), nullable=False),
        StructField("y", IntegerType(), nullable=False),
        StructField("z", IntegerType(), nullable=False),
        StructField("t1", FloatType(), nullable=False),
        StructField("flair", FloatType(), nullable=False),
        StructField("t1_flair_ratio", FloatType(), nullable=False),
        StructField("spatial_prior", FloatType(), nullable=False),
        StructField("t1_zscore", FloatType(), nullable=False),
        StructField("flair_zscore", FloatType(), nullable=False),
        StructField("t1_local_mean", FloatType(), nullable=False),
        StructField("t1_local_std", FloatType(), nullable=False),
        StructField("flair_local_mean", FloatType(), nullable=False),
        StructField("flair_local_std", FloatType(), nullable=False),
        StructField("x_norm", FloatType(), nullable=False),
        StructField("y_norm", FloatType(), nullable=False),
        StructField("z_norm", FloatType(), nullable=False),
        StructField("distance_to_center", FloatType(), nullable=False),
    ]
    if include_shape_columns:
        fields.extend(
            [
                StructField("shape_z", IntegerType(), nullable=False),
                StructField("shape_y", IntegerType(), nullable=False),
                StructField("shape_x", IntegerType(), nullable=False),
            ]
        )
    if include_label:
        fields.append(StructField("label", IntegerType(), nullable=True))
    return StructType(fields)


def _compute_feature_arrays(
    t1: np.ndarray,
    flair: np.ndarray,
    spatial_prior: np.ndarray,
    ratio_epsilon: float,
) -> dict[str, np.ndarray]:
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
    return {
        "mask": mask,
        "x": x_idx,
        "y": y_idx,
        "z": z_idx,
        "t1": t1,
        "flair": flair,
        "t1_flair_ratio": ratio,
        "spatial_prior": spatial_prior.astype(np.float32, copy=False),
        "t1_zscore": t1_zscore,
        "flair_zscore": flair_zscore,
        "t1_local_mean": t1_local_mean,
        "t1_local_std": t1_local_std,
        "flair_local_mean": flair_local_mean,
        "flair_local_std": flair_local_std,
        "x_norm": x_norm.astype(np.float32, copy=False),
        "y_norm": y_norm.astype(np.float32, copy=False),
        "z_norm": z_norm.astype(np.float32, copy=False),
        "distance_to_center": center_distance,
    }


def _selected_voxel_mask(mask: np.ndarray, sparse_brain_only: bool) -> np.ndarray:
    if sparse_brain_only:
        return mask.ravel().astype(bool, copy=False)
    return np.ones(mask.size, dtype=bool)


def _iter_feature_rows(
    subject_id: str,
    feature_arrays: dict[str, np.ndarray],
    *,
    label_col: Optional[np.ndarray],
    sparse_brain_only: bool,
    include_shape_columns: bool,
    include_label: bool,
) -> Iterator[tuple]:
    selected = _selected_voxel_mask(feature_arrays["mask"], sparse_brain_only)
    shape = feature_arrays["t1"].shape

    columns = (
        feature_arrays["x"].ravel()[selected],
        feature_arrays["y"].ravel()[selected],
        feature_arrays["z"].ravel()[selected],
        feature_arrays["t1"].ravel()[selected],
        feature_arrays["flair"].ravel()[selected],
        feature_arrays["t1_flair_ratio"].ravel()[selected],
        feature_arrays["spatial_prior"].ravel()[selected],
        feature_arrays["t1_zscore"].ravel()[selected],
        feature_arrays["flair_zscore"].ravel()[selected],
        feature_arrays["t1_local_mean"].ravel()[selected],
        feature_arrays["t1_local_std"].ravel()[selected],
        feature_arrays["flair_local_mean"].ravel()[selected],
        feature_arrays["flair_local_std"].ravel()[selected],
        feature_arrays["x_norm"].ravel()[selected],
        feature_arrays["y_norm"].ravel()[selected],
        feature_arrays["z_norm"].ravel()[selected],
        feature_arrays["distance_to_center"].ravel()[selected],
    )
    labels = None
    if include_label:
        labels = label_col.ravel()[selected] if label_col is not None else np.full(selected.sum(), None)

    for idx, values in enumerate(zip(*columns)):
        row = (
            subject_id,
            int(values[0]),
            int(values[1]),
            int(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5]),
            float(values[6]),
            float(values[7]),
            float(values[8]),
            float(values[9]),
            float(values[10]),
            float(values[11]),
            float(values[12]),
            float(values[13]),
            float(values[14]),
            float(values[15]),
            float(values[16]),
        )
        if include_shape_columns:
            row += (int(shape[0]), int(shape[1]), int(shape[2]))
        if include_label:
            label_value = None if labels is None else labels[idx]
            row += (None if label_value is None else int(label_value),)
        yield row


def _feature_rows_for_tasks(
    tasks: Iterable[SubjectFeatureTask],
    *,
    ratio_epsilon: float,
    sparse_brain_only: bool,
    include_shape_columns: bool,
    include_label: bool,
) -> Iterator[tuple]:
    for task in tasks:
        t0 = time.time()
        t1, flair, subject_affine = validate_subject_volumes(task.t1_path, task.flair_path)
        spatial_prior = validate_spatial_prior_template(
            subject_shape=t1.shape,
            subject_affine=subject_affine,
            spatial_prior_path=task.spatial_prior_path,
        )
        label_col = validate_binary_mask(task.mask_path).astype(np.int32) if task.mask_path else None
        feature_arrays = _compute_feature_arrays(t1, flair, spatial_prior, ratio_epsilon)
        row_count = int(feature_arrays["mask"].sum()) if sparse_brain_only else int(t1.size)
        logger.info(
            "build_feature_rows: subject=%s voxels=%d sparse=%s emitted_rows=%d elapsed=%.2fs",
            task.subject_id,
            t1.size,
            sparse_brain_only,
            row_count,
            time.time() - t0,
        )
        yield from _iter_feature_rows(
            task.subject_id,
            feature_arrays,
            label_col=label_col,
            sparse_brain_only=sparse_brain_only,
            include_shape_columns=include_shape_columns,
            include_label=include_label,
        )


def build_feature_dataframe_for_tasks(
    spark: "SparkSession",
    tasks: Sequence[SubjectFeatureTask],
    *,
    subject_parallelism: int = 0,
    ratio_epsilon: float = 1e-6,
    sparse_brain_only: bool = True,
    include_shape_columns: bool = False,
    include_label: Optional[bool] = None,
) -> "DataFrame":
    """Build one distributed feature DataFrame for multiple subjects."""
    if not tasks:
        raise ValueError("tasks must not be empty")
    include_label = any(task.mask_path is not None for task in tasks) if include_label is None else include_label
    parallelism = (
        subject_parallelism
        if subject_parallelism > 0
        else max(1, min(len(tasks), spark.sparkContext.defaultParallelism))
    )
    logger.info(
        "building distributed feature DataFrame: subjects=%d subject_parallelism=%d sparse=%s",
        len(tasks),
        parallelism,
        sparse_brain_only,
    )
    schema = feature_row_schema(
        include_label=include_label,
        include_shape_columns=include_shape_columns,
    )
    rdd = spark.sparkContext.parallelize(list(tasks), parallelism).mapPartitions(
        lambda part: _feature_rows_for_tasks(
            part,
            ratio_epsilon=ratio_epsilon,
            sparse_brain_only=sparse_brain_only,
            include_shape_columns=include_shape_columns,
            include_label=include_label,
        )
    )
    return spark.createDataFrame(rdd, schema=schema)


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
    sparse_brain_only: bool = False,
    include_shape_columns: bool = False,
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

    feature_arrays = _compute_feature_arrays(t1, flair, spatial_prior, ratio_epsilon)
    selected = _selected_voxel_mask(feature_arrays["mask"], sparse_brain_only)
    selected_count = int(selected.sum())
    if selected_count == 0:
        result = spark.createDataFrame(
            [],
            schema=feature_row_schema(
                include_label=label_col is not None,
                include_shape_columns=include_shape_columns,
            ),
        ).repartition(plan.partition_count)
        logger.info(
            "build_feature_dataframe: subject=%s voxels=%d partitions=%d emitted_rows=0 elapsed=%.2fs",
            subject_id,
            t1.size,
            plan.partition_count,
            time.time() - t0,
        )
        return result

    pdf = pd.DataFrame(
        {
            "subject_id": subject_id,
            "x": feature_arrays["x"].ravel()[selected],
            "y": feature_arrays["y"].ravel()[selected],
            "z": feature_arrays["z"].ravel()[selected],
            "t1": feature_arrays["t1"].ravel()[selected],
            "flair": feature_arrays["flair"].ravel()[selected],
            "t1_flair_ratio": feature_arrays["t1_flair_ratio"].ravel()[selected],
            "spatial_prior": feature_arrays["spatial_prior"].ravel()[selected],
            "t1_zscore": feature_arrays["t1_zscore"].ravel()[selected],
            "flair_zscore": feature_arrays["flair_zscore"].ravel()[selected],
            "t1_local_mean": feature_arrays["t1_local_mean"].ravel()[selected],
            "t1_local_std": feature_arrays["t1_local_std"].ravel()[selected],
            "flair_local_mean": feature_arrays["flair_local_mean"].ravel()[selected],
            "flair_local_std": feature_arrays["flair_local_std"].ravel()[selected],
            "x_norm": feature_arrays["x_norm"].ravel()[selected],
            "y_norm": feature_arrays["y_norm"].ravel()[selected],
            "z_norm": feature_arrays["z_norm"].ravel()[selected],
            "distance_to_center": feature_arrays["distance_to_center"].ravel()[selected],
        }
    )
    if include_shape_columns:
        pdf["shape_z"] = int(t1.shape[0])
        pdf["shape_y"] = int(t1.shape[1])
        pdf["shape_x"] = int(t1.shape[2])
    if label_col is not None:
        pdf["label"] = label_col.ravel()[selected]

    result = spark.createDataFrame(pdf).repartition(plan.partition_count)
    logger.info(
        "build_feature_dataframe: subject=%s voxels=%d partitions=%d emitted_rows=%d sparse=%s elapsed=%.2fs",
        subject_id,
        t1.size,
        plan.partition_count,
        len(pdf),
        sparse_brain_only,
        time.time() - t0,
    )
    return result
