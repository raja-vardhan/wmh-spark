"""Shared pytest fixtures.

Synthetic volumes are sized (24, 32, 28) — small enough to run the full
test suite in seconds without a real MRI dataset or a multi-node cluster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wmh_spark.io_utils import save_volume


@pytest.fixture(scope="session")
def spark_session():
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName("wmh-spark-test")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def synthetic_volume_shape():
    return (24, 32, 28)


@pytest.fixture(scope="session")
def synthetic_affine():
    a = np.eye(4, dtype=np.float32) * 2.0
    a[3, 3] = 1.0
    a[:3, 3] = [-24.0, -32.0, -28.0]
    return a


def _make_brain(shape):
    z, y, x = np.indices(shape, dtype=np.float32)
    cz, cy, cx = (s / 2 for s in shape)
    rz, ry, rx = (s / 2.5 for s in shape)
    ellipsoid = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    brain = ellipsoid < 1.0
    ventricles = ellipsoid < 0.05
    wm = brain & ~ventricles

    rng = np.random.default_rng(0)
    flair = np.zeros(shape, dtype=np.float32)
    flair[wm] = 100 + rng.normal(0, 5, wm.sum())
    flair[ventricles] = 20

    lesion = np.zeros(shape, dtype=bool)
    lz, ly, lx = (s // 2 - 3 for s in shape)
    lesion[lz : lz + 3, ly : ly + 4, lx : lx + 5] = True
    lesion &= wm
    flair[lesion] = 200

    t1 = np.zeros(shape, dtype=np.float32)
    t1[wm] = 150 + np.random.default_rng(1).normal(0, 5, wm.sum())
    t1[ventricles] = 30
    t1[lesion] = 152

    mask = lesion.astype(np.uint8)
    return flair, t1, mask


@pytest.fixture
def synthetic_subject(tmp_path, synthetic_volume_shape, synthetic_affine):
    flair, t1, mask = _make_brain(synthetic_volume_shape)
    subj = tmp_path / "subj_synth"
    subj.mkdir()
    save_volume(flair, synthetic_affine, subj / "flair.nii.gz")
    save_volume(t1, synthetic_affine, subj / "t1.nii.gz")
    save_volume(mask, synthetic_affine, subj / "wmh_mask.nii.gz", dtype=np.uint8)
    return subj
