"""Tests for distributed voxel feature extraction (Feature 3)."""

from __future__ import annotations

import numpy as np
import pytest
from pyspark.sql import functions as F

from wmh_spark.feature_extraction import (
    FeaturePartitionPlan,
    SubjectFeatureTask,
    build_feature_dataframe,
    build_feature_dataframe_for_tasks,
    derive_feature_partition_count,
)
from wmh_spark.io_utils import load_volume, save_volume


def _write_spatial_prior(path, shape, affine):
    z_idx, y_idx, x_idx = np.indices(shape, dtype=np.float32)
    prior = (x_idx * 0.01 + y_idx * 0.001 + z_idx * 0.0001).astype(np.float32)
    save_volume(prior, affine, path)
    return prior


def test_build_feature_dataframe_schema_ratio_and_spatial_prior(
    spark_session,
    synthetic_subject,
    synthetic_volume_shape,
    synthetic_affine,
):
    prior_path = synthetic_subject / "spatial_prior.nii.gz"
    prior = _write_spatial_prior(prior_path, synthetic_volume_shape, synthetic_affine)

    df = build_feature_dataframe(
        spark=spark_session,
        subject_id="subj_synth",
        t1_path=synthetic_subject / "t1.nii.gz",
        flair_path=synthetic_subject / "flair.nii.gz",
        spatial_prior_path=prior_path,
        partition_count=3,
    )

    for column in [
        "subject_id",
        "x",
        "y",
        "z",
        "t1",
        "flair",
        "t1_flair_ratio",
        "spatial_prior",
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
    ]:
        assert column in df.columns
    assert df.count() == np.prod(synthetic_volume_shape)

    t1, _ = load_volume(synthetic_subject / "t1.nii.gz")
    flair, _ = load_volume(synthetic_subject / "flair.nii.gz")
    z, y, x = np.argwhere(flair > 0)[0].tolist()

    row = (
        df.where((F.col("x") == x) & (F.col("y") == y) & (F.col("z") == z))
        .select(
            "t1",
            "flair",
            "t1_flair_ratio",
            "spatial_prior",
            "t1_zscore",
            "flair_zscore",
            "x_norm",
            "y_norm",
            "z_norm",
            "distance_to_center",
        )
        .first()
    )

    assert row["t1"] == pytest.approx(float(t1[z, y, x]), rel=1e-6)
    assert row["flair"] == pytest.approx(float(flair[z, y, x]), rel=1e-6)
    assert row["t1_flair_ratio"] == pytest.approx(
        float(t1[z, y, x] / flair[z, y, x]),
        rel=1e-6,
    )
    assert row["spatial_prior"] == pytest.approx(float(prior[z, y, x]), rel=1e-6)
    assert 0.0 <= row["x_norm"] <= 1.0
    assert 0.0 <= row["y_norm"] <= 1.0
    assert 0.0 <= row["z_norm"] <= 1.0
    assert 0.0 <= row["distance_to_center"] <= 1.0
    assert np.isfinite(row["t1_zscore"])
    assert np.isfinite(row["flair_zscore"])


def test_feature_ratio_uses_spark_expression_not_python_udf(
    spark_session,
    synthetic_subject,
    synthetic_volume_shape,
    synthetic_affine,
):
    prior_path = synthetic_subject / "spatial_prior.nii.gz"
    _write_spatial_prior(prior_path, synthetic_volume_shape, synthetic_affine)

    df = build_feature_dataframe(
        spark=spark_session,
        subject_id="subj_synth",
        t1_path=synthetic_subject / "t1.nii.gz",
        flair_path=synthetic_subject / "flair.nii.gz",
        spatial_prior_path=prior_path,
        partition_count=2,
    )

    executed_plan = df._jdf.queryExecution().executedPlan().toString()
    assert "BatchEvalPython" not in executed_plan
    assert "PythonUDF" not in executed_plan


def test_spatial_prior_template_shape_mismatch_fails_fast(
    spark_session,
    synthetic_subject,
    synthetic_affine,
):
    prior_path = synthetic_subject / "bad_spatial_prior.nii.gz"
    bad_prior = np.zeros((3, 4, 5), dtype=np.float32)
    save_volume(bad_prior, synthetic_affine, prior_path)

    with pytest.raises(ValueError, match="spatial prior shape"):
        build_feature_dataframe(
            spark=spark_session,
            subject_id="subj_synth",
            t1_path=synthetic_subject / "t1.nii.gz",
            flair_path=synthetic_subject / "flair.nii.gz",
            spatial_prior_path=prior_path,
        )


def test_derive_feature_partition_count_keeps_partitions_under_shuffle_target():
    plan = derive_feature_partition_count(
        voxel_count=1_000_000,
        bytes_per_row=64,
        target_partition_bytes=1 * 1024 * 1024,
        min_partitions=1,
    )

    assert isinstance(plan, FeaturePartitionPlan)
    assert plan.partition_count >= 1
    assert plan.estimated_bytes_per_partition <= 1 * 1024 * 1024
    assert plan.estimated_bytes_per_partition <= 16 * 1024**3

    override = derive_feature_partition_count(
        voxel_count=1_000_000,
        bytes_per_row=64,
        override=7,
    )
    assert override.partition_count == 7


def test_build_feature_dataframe_applies_partition_plan(
    spark_session,
    synthetic_subject,
    synthetic_volume_shape,
    synthetic_affine,
):
    prior_path = synthetic_subject / "spatial_prior.nii.gz"
    _write_spatial_prior(prior_path, synthetic_volume_shape, synthetic_affine)

    df = build_feature_dataframe(
        spark=spark_session,
        subject_id="subj_synth",
        t1_path=synthetic_subject / "t1.nii.gz",
        flair_path=synthetic_subject / "flair.nii.gz",
        spatial_prior_path=prior_path,
        partition_count=5,
    )

    assert spark_session.conf.get("spark.sql.shuffle.partitions") == "5"
    assert df.rdd.getNumPartitions() == 5


def test_build_feature_dataframe_sparse_retains_only_brain_voxels(
    spark_session,
    synthetic_subject,
    synthetic_volume_shape,
    synthetic_affine,
):
    prior_path = synthetic_subject / "spatial_prior.nii.gz"
    _write_spatial_prior(prior_path, synthetic_volume_shape, synthetic_affine)

    dense = build_feature_dataframe(
        spark=spark_session,
        subject_id="subj_synth",
        t1_path=synthetic_subject / "t1.nii.gz",
        flair_path=synthetic_subject / "flair.nii.gz",
        spatial_prior_path=prior_path,
        partition_count=2,
    )
    sparse = build_feature_dataframe(
        spark=spark_session,
        subject_id="subj_synth",
        t1_path=synthetic_subject / "t1.nii.gz",
        flair_path=synthetic_subject / "flair.nii.gz",
        spatial_prior_path=prior_path,
        partition_count=2,
        sparse_brain_only=True,
        include_shape_columns=True,
    )

    dense_brain = dense.where((F.abs(F.col("t1")) > 1e-6) | (F.abs(F.col("flair")) > 1e-6))
    assert sparse.count() == dense_brain.count()
    assert sparse.count() < dense.count()
    assert {"shape_z", "shape_y", "shape_x"} <= set(sparse.columns)

    coord = dense_brain.select("x", "y", "z").first()
    dense_row = (
        dense.where(
            (F.col("x") == coord["x"])
            & (F.col("y") == coord["y"])
            & (F.col("z") == coord["z"])
        )
        .select("t1", "flair", "spatial_prior", "distance_to_center")
        .first()
    )
    sparse_row = (
        sparse.where(
            (F.col("x") == coord["x"])
            & (F.col("y") == coord["y"])
            & (F.col("z") == coord["z"])
        )
        .select("t1", "flair", "spatial_prior", "distance_to_center")
        .first()
    )
    assert sparse_row == dense_row


def test_build_feature_dataframe_for_tasks_combines_sparse_subjects(
    spark_session,
    synthetic_subject,
    synthetic_volume_shape,
    synthetic_affine,
    tmp_path,
):
    prior_a = synthetic_subject / "spatial_prior_a.nii.gz"
    _write_spatial_prior(prior_a, synthetic_volume_shape, synthetic_affine)

    subject_b = tmp_path / "subj_b"
    subject_b.mkdir()
    for name in ("t1.nii.gz", "flair.nii.gz", "wmh_mask.nii.gz"):
        data, affine = load_volume(synthetic_subject / name)
        save_volume(data, affine, subject_b / name, dtype=np.uint8 if "mask" in name else np.float32)
    prior_b = subject_b / "spatial_prior_b.nii.gz"
    _write_spatial_prior(prior_b, synthetic_volume_shape, synthetic_affine)

    tasks = [
        SubjectFeatureTask(
            subject_id="subj_a",
            t1_path=str(synthetic_subject / "t1.nii.gz"),
            flair_path=str(synthetic_subject / "flair.nii.gz"),
            spatial_prior_path=str(prior_a),
            mask_path=str(synthetic_subject / "wmh_mask.nii.gz"),
        ),
        SubjectFeatureTask(
            subject_id="subj_b",
            t1_path=str(subject_b / "t1.nii.gz"),
            flair_path=str(subject_b / "flair.nii.gz"),
            spatial_prior_path=str(prior_b),
            mask_path=str(subject_b / "wmh_mask.nii.gz"),
        ),
    ]

    df = build_feature_dataframe_for_tasks(
        spark_session,
        tasks,
        subject_parallelism=2,
        sparse_brain_only=True,
        include_shape_columns=True,
        include_label=True,
    )

    assert {row["subject_id"] for row in df.select("subject_id").distinct().collect()} == {"subj_a", "subj_b"}
    assert {"shape_z", "shape_y", "shape_x", "label"} <= set(df.columns)
    assert df.where(F.col("label") == 1).count() > 0
