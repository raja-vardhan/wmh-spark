"""Tests for 3D connected-component post-processing."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pyspark.sql import functions as F

from wmh_spark.postprocessing.connected_components import (
    PostProcessingConfig,
    component_size_map,
    label_connected_components,
    postprocess_predictions,
    reconstruct_prediction_volume,
    summarize_subject_predictions,
)


def _prediction_df(spark_session, shape, positive_coords):
    positives = set(positive_coords)
    rows = []
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                rows.append(
                    {
                        "subject_id": "subj",
                        "x": x,
                        "y": y,
                        "z": z,
                        "predicted_mask": 1 if (z, y, x) in positives else 0,
                    }
                )
    return spark_session.createDataFrame(rows).repartition(2)


def _prediction_df_with_probability(spark_session, shape, probabilities):
    rows = []
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                probability = probabilities.get((z, y, x), 0.0)
                rows.append(
                    {
                        "subject_id": "subj",
                        "x": x,
                        "y": y,
                        "z": z,
                        "predicted_mask": 1 if probability > 0 else 0,
                        "wmh_probability": float(probability),
                        "spatial_prior": 0.8 if probability > 0 else 0.1,
                    }
                )
    return spark_session.createDataFrame(rows).repartition(2)


def test_reconstruct_prediction_volume_from_flat_dataframe(spark_session):
    shape = (3, 4, 5)
    positives = {(0, 1, 2), (2, 3, 4)}
    df = _prediction_df(spark_session, shape, positives)

    volume = reconstruct_prediction_volume(
        df,
        PostProcessingConfig(volume_shape=shape),
    )

    assert volume.shape == shape
    assert volume.dtype == np.uint8
    assert volume[0, 1, 2] == 1
    assert volume[2, 3, 4] == 1
    assert int(volume.sum()) == 2


def test_connected_components_uses_26_connectivity():
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[0, 0, 0] = 1
    volume[1, 1, 1] = 1  # corner-touching with (0, 0, 0)
    volume[3, 3, 3] = 1

    labeled, count = label_connected_components(volume)

    assert count == 2
    assert labeled[0, 0, 0] == labeled[1, 1, 1]
    assert labeled[3, 3, 3] != labeled[0, 0, 0]


def test_component_sizes_are_calculated():
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[0, 0, 0] = 1
    volume[0, 0, 1] = 1
    volume[0, 1, 1] = 1
    volume[3, 3, 3] = 1

    labeled, _ = label_connected_components(volume)
    sizes = component_size_map(labeled)

    assert sorted(sizes.values()) == [1, 3]


def test_postprocess_predictions_filters_small_clusters_and_preserves_rows(
    spark_session,
):
    shape = (4, 4, 4)
    large_cluster = {(0, 0, 0), (0, 0, 1), (0, 1, 1)}
    tiny_cluster = {(3, 3, 3)}
    df = _prediction_df(spark_session, shape, large_cluster | tiny_cluster)
    config = PostProcessingConfig(volume_shape=shape, min_cluster_size=2)

    result = postprocess_predictions(df, config)

    assert result.count() == df.count()
    for column in ["component_id", "cluster_size", "postprocessed_mask"]:
        assert column in result.columns

    kept = {
        (row["z"], row["y"], row["x"])
        for row in result.where(F.col("postprocessed_mask") == 1).select("z", "y", "x").collect()
    }
    removed_row = (
        result.where((F.col("z") == 3) & (F.col("y") == 3) & (F.col("x") == 3))
        .select("predicted_mask", "postprocessed_mask", "cluster_size")
        .first()
    )

    assert kept == large_cluster
    assert removed_row["predicted_mask"] == 1
    assert removed_row["postprocessed_mask"] == 0
    assert removed_row["cluster_size"] == 1


def test_postprocess_predictions_requires_prediction_column(spark_session):
    df = spark_session.createDataFrame([{"x": 0, "y": 0, "z": 0}])

    with pytest.raises(ValueError, match="predicted_mask"):
        postprocess_predictions(df, PostProcessingConfig(volume_shape=(1, 1, 1)))


def test_postprocess_predictions_filters_low_confidence_components(spark_session):
    shape = (3, 3, 3)
    probabilities = {
        (0, 0, 0): 0.9,
        (0, 0, 1): 0.9,
        (2, 2, 2): 0.2,
        (2, 2, 1): 0.2,
    }
    df = _prediction_df_with_probability(spark_session, shape, probabilities)

    result = postprocess_predictions(
        df,
        PostProcessingConfig(
            volume_shape=shape,
            min_cluster_size=2,
            min_component_mean_probability=0.5,
            min_component_peak_probability=0.5,
        ),
    )

    kept = {
        (row["z"], row["y"], row["x"])
        for row in result.where(F.col("postprocessed_mask") == 1).select("z", "y", "x").collect()
    }
    assert kept == {(0, 0, 0), (0, 0, 1)}


def test_summarize_subject_predictions_compacts_sparse_rows(spark_session):
    shape = (4, 4, 4)
    large_cluster = {(0, 0, 0), (0, 0, 1), (0, 1, 1)}
    tiny_cluster = {(3, 3, 3)}
    rows = []
    for z, y, x in sorted(large_cluster | tiny_cluster):
        rows.append(
            {
                "subject_id": "subj",
                "shape_z": shape[0],
                "shape_y": shape[1],
                "shape_x": shape[2],
                "z": z,
                "y": y,
                "x": x,
                "predicted_mask": 1,
                "wmh_probability": 0.9,
                "spatial_prior": 0.8,
            }
        )
    df = spark_session.createDataFrame(rows).repartition(2)

    summary = summarize_subject_predictions(
        df,
        PostProcessingConfig(min_cluster_size=2),
    ).first()

    kept = {tuple(coord) for coord in json.loads(summary["positive_xyz_json"])}
    assert kept == large_cluster
    assert summary["raw_predicted_voxels"] == 4
    assert summary["predicted_voxels"] == 3
    assert summary["component_count"] == 2
