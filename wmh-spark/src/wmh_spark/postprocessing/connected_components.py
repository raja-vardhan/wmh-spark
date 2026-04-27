"""3D connected-component post-processing for raw voxel predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from scipy import ndimage

if TYPE_CHECKING:
    from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)

COORDINATE_COLUMNS = ("x", "y", "z")


@dataclass(frozen=True)
class PostProcessingConfig:
    """Settings for volumetric connected-component filtering."""

    prediction_column: str = "predicted_mask"
    component_column: str = "component_id"
    cluster_size_column: str = "cluster_size"
    output_column: str = "postprocessed_mask"
    min_cluster_size: int = 10
    volume_shape: tuple[int, int, int] | None = None


def _validate_required_columns(df: "DataFrame", config: PostProcessingConfig) -> None:
    required = [*COORDINATE_COLUMNS, config.prediction_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required post-processing columns: {missing}")


def _validate_config(config: PostProcessingConfig) -> None:
    if config.min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive")
    if config.volume_shape is not None:
        if len(config.volume_shape) != 3:
            raise ValueError("volume_shape must be a (z, y, x) tuple")
        if any(dim <= 0 for dim in config.volume_shape):
            raise ValueError("volume_shape dimensions must be positive")


def infer_volume_shape(df: "DataFrame") -> tuple[int, int, int]:
    """Infer `(z, y, x)` volume shape from max DataFrame coordinates."""
    missing = [column for column in COORDINATE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"missing coordinate columns: {missing}")

    row = df.selectExpr("max(z) as max_z", "max(y) as max_y", "max(x) as max_x").first()
    if row is None or row["max_z"] is None or row["max_y"] is None or row["max_x"] is None:
        raise ValueError("cannot infer volume shape from an empty DataFrame")

    return (int(row["max_z"]) + 1, int(row["max_y"]) + 1, int(row["max_x"]) + 1)


def _collect_coordinate_predictions(
    df: "DataFrame",
    config: PostProcessingConfig,
) -> list[tuple[int, int, int, int]]:
    rows = (
        df.select("z", "y", "x", config.prediction_column)
        .where(df[config.prediction_column].isNotNull())
        .collect()
    )
    collected: list[tuple[int, int, int, int]] = []
    for row in rows:
        z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
        if z < 0 or y < 0 or x < 0:
            raise ValueError(f"voxel coordinates must be non-negative: {(z, y, x)}")
        prediction = 1 if int(row[config.prediction_column]) != 0 else 0
        collected.append((z, y, x, prediction))
    return collected


def reconstruct_prediction_volume(
    df: "DataFrame",
    config: PostProcessingConfig | None = None,
) -> np.ndarray:
    """Reconstruct flattened predictions into a binary `(z, y, x)` volume."""
    config = config or PostProcessingConfig()
    _validate_config(config)
    _validate_required_columns(df, config)

    shape = config.volume_shape or infer_volume_shape(df)
    volume = np.zeros(shape, dtype=np.uint8)

    for z, y, x, prediction in _collect_coordinate_predictions(df, config):
        if z >= shape[0] or y >= shape[1] or x >= shape[2]:
            raise ValueError(
                f"voxel coordinate {(z, y, x)} exceeds volume_shape {shape}"
            )
        volume[z, y, x] = prediction

    return volume


def label_connected_components(volume: np.ndarray) -> tuple[np.ndarray, int]:
    """Label positive voxels with 26-connectivity."""
    if volume.ndim != 3:
        raise ValueError("connected component labeling requires a 3D volume")

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, component_count = ndimage.label(volume.astype(bool), structure=structure)
    return labeled.astype(np.int32), int(component_count)


def component_size_map(labeled_volume: np.ndarray) -> dict[int, int]:
    """Return `{component_id: voxel_count}` excluding background component 0."""
    if labeled_volume.ndim != 3:
        raise ValueError("component sizes require a 3D labeled volume")

    counts = np.bincount(labeled_volume.ravel())
    return {
        component_id: int(count)
        for component_id, count in enumerate(counts)
        if component_id != 0 and count > 0
    }


def filter_components_by_size(
    labeled_volume: np.ndarray,
    sizes: dict[int, int],
    min_cluster_size: int,
) -> np.ndarray:
    """Return a binary mask retaining only components at or above threshold."""
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive")

    keep_ids = [component_id for component_id, size in sizes.items() if size >= min_cluster_size]
    if not keep_ids:
        return np.zeros(labeled_volume.shape, dtype=np.uint8)
    return np.isin(labeled_volume, keep_ids).astype(np.uint8)


def _component_rows(
    labeled_volume: np.ndarray,
    filtered_volume: np.ndarray,
    sizes: dict[int, int],
    config: PostProcessingConfig,
) -> list[dict[str, int]]:
    z_idx, y_idx, x_idx = np.indices(labeled_volume.shape, dtype=np.int32)
    component_ids = labeled_volume.ravel()

    rows: list[dict[str, int]] = []
    for z, y, x, component_id, filtered_value in zip(
        z_idx.ravel(),
        y_idx.ravel(),
        x_idx.ravel(),
        component_ids,
        filtered_volume.ravel(),
    ):
        component_id = int(component_id)
        rows.append(
            {
                "z": int(z),
                "y": int(y),
                "x": int(x),
                config.component_column: component_id,
                config.cluster_size_column: int(sizes.get(component_id, 0)),
                config.output_column: int(filtered_value),
            }
        )
    return rows


def postprocess_predictions(
    df: "DataFrame",
    config: PostProcessingConfig | None = None,
) -> "DataFrame":
    """Append component labels, cluster sizes, and thresholded mask values."""
    config = config or PostProcessingConfig()
    _validate_config(config)
    volume = reconstruct_prediction_volume(df, config)
    labeled, component_count = label_connected_components(volume)
    sizes = component_size_map(labeled)
    filtered = filter_components_by_size(labeled, sizes, config.min_cluster_size)

    spark = df.sparkSession
    component_df = spark.createDataFrame(_component_rows(labeled, filtered, sizes, config))
    result = df.join(component_df, on=["x", "y", "z"], how="left")

    logger.info(
        "postprocess_predictions: components=%d retained_voxels=%d min_cluster_size=%d",
        component_count,
        int(filtered.sum()),
        config.min_cluster_size,
    )
    return result
