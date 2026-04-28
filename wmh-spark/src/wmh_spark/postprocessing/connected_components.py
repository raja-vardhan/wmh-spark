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
    probability_column: str = "wmh_probability"
    spatial_prior_column: str = "spatial_prior"
    component_column: str = "component_id"
    cluster_size_column: str = "cluster_size"
    mean_probability_column: str = "component_mean_probability"
    peak_probability_column: str = "component_peak_probability"
    mean_spatial_prior_column: str = "component_mean_spatial_prior"
    output_column: str = "postprocessed_mask"
    min_cluster_size: int = 10
    min_component_mean_probability: float = 0.0
    min_component_peak_probability: float = 0.0
    anatomical_gate_enabled: bool = False
    min_component_mean_spatial_prior: float = 0.0
    volume_shape: tuple[int, int, int] | None = None


def _validate_required_columns(df: "DataFrame", config: PostProcessingConfig) -> None:
    required = [*COORDINATE_COLUMNS, config.prediction_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required post-processing columns: {missing}")


def _validate_config(config: PostProcessingConfig) -> None:
    if config.min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive")
    for name, value in (
        ("min_component_mean_probability", config.min_component_mean_probability),
        ("min_component_peak_probability", config.min_component_peak_probability),
        ("min_component_mean_spatial_prior", config.min_component_mean_spatial_prior),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
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


def _collect_coordinate_rows(
    df: "DataFrame",
    columns: Sequence[str],
) -> list:
    rows = (
        df.select("z", "y", "x", *columns)
        .where(df[columns[0]].isNotNull())
        .collect()
    )
    collected: list = []
    for row in rows:
        z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
        if z < 0 or y < 0 or x < 0:
            raise ValueError(f"voxel coordinates must be non-negative: {(z, y, x)}")
        collected.append(row)
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

    for row in _collect_coordinate_rows(df, [config.prediction_column]):
        z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
        prediction = 1 if int(row[config.prediction_column]) != 0 else 0
        if z >= shape[0] or y >= shape[1] or x >= shape[2]:
            raise ValueError(
                f"voxel coordinate {(z, y, x)} exceeds volume_shape {shape}"
            )
        volume[z, y, x] = prediction

    return volume


def reconstruct_scalar_volume(
    df: "DataFrame",
    column: str,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Reconstruct a scalar `(z, y, x)` volume from a DataFrame column."""
    if column not in df.columns:
        raise ValueError(f"missing required scalar column: {column}")
    volume = np.zeros(shape, dtype=np.float32)
    for row in _collect_coordinate_rows(df, [column]):
        z, y, x = int(row["z"]), int(row["y"]), int(row["x"])
        volume[z, y, x] = float(row[column] or 0.0)
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


def _component_stat_map(
    labeled_volume: np.ndarray,
    value_volume: np.ndarray,
    reducer: str,
) -> dict[int, float]:
    component_ids = [component_id for component_id in np.unique(labeled_volume) if component_id != 0]
    if not component_ids:
        return {}
    if reducer == "mean":
        values = ndimage.mean(value_volume, labels=labeled_volume, index=component_ids)
    elif reducer == "max":
        values = ndimage.maximum(value_volume, labels=labeled_volume, index=component_ids)
    else:
        raise ValueError(f"unsupported reducer: {reducer}")
    return {
        int(component_id): float(value)
        for component_id, value in zip(component_ids, values)
    }


def filter_components(
    labeled_volume: np.ndarray,
    sizes: dict[int, int],
    mean_probabilities: dict[int, float],
    peak_probabilities: dict[int, float],
    mean_spatial_priors: dict[int, float],
    config: PostProcessingConfig,
) -> np.ndarray:
    """Retain components that satisfy size, confidence, and optional anatomy gates."""
    keep_ids: list[int] = []
    for component_id, size in sizes.items():
        if size < config.min_cluster_size:
            continue
        if mean_probabilities.get(component_id, 0.0) < float(config.min_component_mean_probability):
            continue
        if peak_probabilities.get(component_id, 0.0) < float(config.min_component_peak_probability):
            continue
        if (
            config.anatomical_gate_enabled
            and mean_spatial_priors.get(component_id, 0.0)
            < float(config.min_component_mean_spatial_prior)
        ):
            continue
        keep_ids.append(component_id)
    if not keep_ids:
        return np.zeros(labeled_volume.shape, dtype=np.uint8)
    return np.isin(labeled_volume, keep_ids).astype(np.uint8)


def _component_rows(
    labeled_volume: np.ndarray,
    filtered_volume: np.ndarray,
    sizes: dict[int, int],
    mean_probabilities: dict[int, float],
    peak_probabilities: dict[int, float],
    mean_spatial_priors: dict[int, float],
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
                config.mean_probability_column: float(mean_probabilities.get(component_id, 0.0)),
                config.peak_probability_column: float(peak_probabilities.get(component_id, 0.0)),
                config.mean_spatial_prior_column: float(
                    mean_spatial_priors.get(component_id, 0.0)
                ),
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
    mean_probabilities: dict[int, float] = {}
    peak_probabilities: dict[int, float] = {}
    mean_spatial_priors: dict[int, float] = {}
    if sizes:
        probability_required = (
            config.min_component_mean_probability > 0
            or config.min_component_peak_probability > 0
        )
        if config.probability_column in df.columns:
            probability_volume = reconstruct_scalar_volume(
                df,
                config.probability_column,
                volume.shape,
            )
            mean_probabilities = _component_stat_map(labeled, probability_volume, reducer="mean")
            peak_probabilities = _component_stat_map(labeled, probability_volume, reducer="max")
        elif probability_required:
            raise ValueError(f"missing required scalar column: {config.probability_column}")

        if config.anatomical_gate_enabled or config.min_component_mean_spatial_prior > 0:
            if config.spatial_prior_column not in df.columns:
                raise ValueError(f"missing required scalar column: {config.spatial_prior_column}")
            spatial_prior_volume = reconstruct_scalar_volume(
                df,
                config.spatial_prior_column,
                volume.shape,
            )
            mean_spatial_priors = _component_stat_map(
                labeled,
                spatial_prior_volume,
                reducer="mean",
            )
    filtered = filter_components(
        labeled,
        sizes,
        mean_probabilities,
        peak_probabilities,
        mean_spatial_priors,
        config,
    )

    spark = df.sparkSession
    component_df = spark.createDataFrame(
        _component_rows(
            labeled,
            filtered,
            sizes,
            mean_probabilities,
            peak_probabilities,
            mean_spatial_priors,
            config,
        )
    )
    result = df.join(component_df, on=["x", "y", "z"], how="left")

    logger.info(
        (
            "postprocess_predictions: components=%d retained_voxels=%d "
            "min_cluster_size=%d min_mean_prob=%.3f min_peak_prob=%.3f anatomy_gate=%s"
        ),
        component_count,
        int(filtered.sum()),
        config.min_cluster_size,
        config.min_component_mean_probability,
        config.min_component_peak_probability,
        config.anatomical_gate_enabled,
    )
    return result
