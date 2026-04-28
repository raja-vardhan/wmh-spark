"""Tests for the Spark MLlib Random Forest classification engine."""

from __future__ import annotations

import numpy as np
import pytest
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors

from wmh_spark.feature_extraction import build_feature_dataframe
from wmh_spark.io_utils import save_volume
from wmh_spark.models.random_forest import (
    ClassificationConfig,
    append_probability_column,
    calculate_class_balance_stats,
    downsample_negative_examples,
    predict_voxel_mask,
    prepare_training_dataframe,
    select_prediction_threshold,
    train_random_forest_model,
)


def _write_spatial_prior(path, shape, affine):
    z_idx, y_idx, x_idx = np.indices(shape, dtype=np.float32)
    prior = (x_idx + y_idx + z_idx).astype(np.float32) / max(shape)
    save_volume(prior, affine, path)


def _rich_feature_row(
    *,
    t1: float,
    flair: float,
    spatial_prior: float,
    x: int = 0,
    y: int = 0,
    z: int = 0,
    label: int | None = None,
    probability=None,
):
    row = {
        "x": x,
        "y": y,
        "z": z,
        "t1": t1,
        "flair": flair,
        "t1_flair_ratio": t1 / flair if flair else 0.0,
        "spatial_prior": spatial_prior,
        "t1_zscore": t1 / 100.0,
        "flair_zscore": flair / 100.0,
        "t1_local_mean": t1 + 1.0,
        "t1_local_std": 0.5,
        "flair_local_mean": flair + 1.0,
        "flair_local_std": 0.5,
        "x_norm": x / 3.0 if x else 0.0,
        "y_norm": y / 3.0 if y else 0.0,
        "z_norm": z / 3.0 if z else 0.0,
        "distance_to_center": 0.5,
    }
    if label is not None:
        row["label"] = label
    if probability is not None:
        row["probability"] = probability
    return row


def _small_labeled_feature_df(spark_session):
    rows = []
    for i in range(48):
        label = 1 if i % 6 == 0 else 0
        flair = float(20 + i)
        t1 = float(10 + (i * 2))
        row = _rich_feature_row(
            x=i % 4,
            y=(i // 4) % 4,
            z=i // 16,
            t1=t1,
            flair=flair,
            spatial_prior=float(label) * 0.8 + 0.05,
            label=label,
        )
        row["subject_id"] = "tiny"
        rows.append(row)
    return spark_session.createDataFrame(rows).repartition(4)


class _StaticProbabilityModel:
    def transform(self, df):
        return df


def test_prepare_training_dataframe_uses_feature3_columns_and_partitions(
    spark_session,
    synthetic_subject,
    synthetic_volume_shape,
    synthetic_affine,
):
    prior_path = synthetic_subject / "spatial_prior.nii.gz"
    _write_spatial_prior(prior_path, synthetic_volume_shape, synthetic_affine)
    feature_df = build_feature_dataframe(
        spark=spark_session,
        subject_id="subj_synth",
        t1_path=synthetic_subject / "t1.nii.gz",
        flair_path=synthetic_subject / "flair.nii.gz",
        spatial_prior_path=prior_path,
        mask_path=synthetic_subject / "wmh_mask.nii.gz",
        partition_count=2,
    )
    config = ClassificationConfig(training_partitions=4)

    prepared = prepare_training_dataframe(feature_df, config)
    row = prepared.select("features", "class_weight").where("label = 1").first()

    assert prepared.rdd.getNumPartitions() == 4
    assert spark_session.conf.get("spark.sql.shuffle.partitions") == "4"
    assert row["features"].size == len(config.feature_columns)
    assert row["class_weight"] > 1.0


def test_class_balance_stats_caps_positive_weight(spark_session):
    df = _small_labeled_feature_df(spark_session)
    config = ClassificationConfig(positive_class_weight_cap=3.0)

    stats = calculate_class_balance_stats(df, config)

    assert stats.negative_count == 40
    assert stats.positive_count == 8
    assert stats.positive_class_weight == pytest.approx(3.0)


def test_train_random_forest_uses_spark_mllib_model(spark_session):
    df = _small_labeled_feature_df(spark_session)
    config = ClassificationConfig(num_trees=5, max_depth=3, training_partitions=4)

    model = train_random_forest_model(df, config)

    assert isinstance(model.stages[-1], RandomForestClassificationModel)
    assert model.stages[-1].getNumTrees == config.num_trees
    assert model.stages[-1].getWeightCol() == config.weight_column


def test_predict_voxel_mask_outputs_binary_column_and_preserves_rows(spark_session):
    df = _small_labeled_feature_df(spark_session)
    config = ClassificationConfig(num_trees=5, max_depth=3, training_partitions=4)
    model = train_random_forest_model(df, config)

    predicted = predict_voxel_mask(model, df, config)
    values = {row["predicted_mask"] for row in predicted.select("predicted_mask").distinct().collect()}

    assert predicted.count() == df.count()
    assert values <= {0, 1}
    assert "wmh_probability" in predicted.columns
    assert "predicted_mask" in predicted.columns
    assert predicted.schema["wmh_probability"].dataType.simpleString() == "double"
    assert predicted.schema["predicted_mask"].dataType.simpleString() == "int"


def test_predict_voxel_mask_thresholds_wmh_probability(spark_session):
    rows = [
        _rich_feature_row(
            t1=1.0,
            flair=2.0,
            spatial_prior=0.1,
            probability=Vectors.dense([0.8, 0.2]),
        ),
        _rich_feature_row(
            t1=2.0,
            flair=3.0,
            spatial_prior=0.9,
            probability=Vectors.dense([0.7, 0.3]),
        ),
    ]
    df = spark_session.createDataFrame(rows)
    config = ClassificationConfig(prediction_threshold=0.25)

    predicted = predict_voxel_mask(_StaticProbabilityModel(), df, config)
    result = predicted.select("wmh_probability", "predicted_mask").collect()

    assert result[0]["wmh_probability"] == pytest.approx(0.2)
    assert result[0]["predicted_mask"] == 0
    assert result[1]["wmh_probability"] == pytest.approx(0.3)
    assert result[1]["predicted_mask"] == 1


def test_training_requires_label_column(spark_session):
    df = _small_labeled_feature_df(spark_session).drop("label")

    with pytest.raises(ValueError, match="label"):
        train_random_forest_model(df, ClassificationConfig())


def test_training_rejects_non_binary_labels(spark_session):
    rows = [
        _rich_feature_row(t1=1.0, flair=2.0, spatial_prior=0.1, label=0),
        _rich_feature_row(t1=2.0, flair=1.0, spatial_prior=0.9, label=2),
    ]
    df = spark_session.createDataFrame(rows)

    with pytest.raises(ValueError, match="binary"):
        train_random_forest_model(df, ClassificationConfig())


def test_training_requires_both_binary_classes(spark_session):
    rows = [
        _rich_feature_row(t1=1.0, flair=2.0, spatial_prior=0.1, label=0),
        _rich_feature_row(t1=2.0, flair=3.0, spatial_prior=0.2, label=0),
    ]
    df = spark_session.createDataFrame(rows)

    with pytest.raises(ValueError, match="both background and WMH"):
        train_random_forest_model(df, ClassificationConfig())


def test_downsample_negative_examples_retains_all_positive_rows(spark_session):
    df = _small_labeled_feature_df(spark_session)
    config = ClassificationConfig(negative_sampling_ratio=1.0, seed=7)

    sampled = downsample_negative_examples(df, config)

    positive_original = df.where("label = 1").count()
    positive_sampled = sampled.where("label = 1").count()
    negative_sampled = sampled.where("label = 0").count()

    assert positive_sampled == positive_original
    assert negative_sampled <= positive_original + 2


def test_select_prediction_threshold_prefers_best_dice_and_lower_fp_burden(spark_session):
    rows = [
        _rich_feature_row(
            t1=1.0,
            flair=2.0,
            spatial_prior=0.2,
            label=1,
            probability=Vectors.dense([0.2, 0.8]),
        ),
        _rich_feature_row(
            t1=1.0,
            flair=2.0,
            spatial_prior=0.2,
            label=0,
            probability=Vectors.dense([0.45, 0.55]),
        ),
        _rich_feature_row(
            t1=1.0,
            flair=2.0,
            spatial_prior=0.2,
            label=0,
            probability=Vectors.dense([0.6, 0.4]),
        ),
    ]
    df = spark_session.createDataFrame(rows)
    config = ClassificationConfig(candidate_thresholds=(0.4, 0.6))

    selection = select_prediction_threshold(append_probability_column(_StaticProbabilityModel(), df, config), config)

    assert selection.threshold == pytest.approx(0.6)
    assert selection.dsc == pytest.approx(1.0)
