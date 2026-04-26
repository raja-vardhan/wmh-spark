"""Tests for io_utils.py — T009-T011, T015-T017, T020-T022, T026a, T033, T034."""

from __future__ import annotations

import numpy as np
import pytest

from wmh_spark.io_utils import (
    SubjectRecord,
    build_voxel_dataframe,
    derive_partition_count,
    load_volume,
    save_volume,
    sha256_of_file,
    validate_binary_mask,
    validate_subject_volumes,
)


# ---------------------------------------------------------------------------
# T009 — dimension mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_validate_subject_volumes_mismatch(tmp_path, synthetic_affine):
    a = np.ones((10, 10, 10), dtype=np.float32)
    b = np.ones((10, 10, 12), dtype=np.float32)
    p1 = tmp_path / "t1.nii.gz"
    p2 = tmp_path / "flair.nii.gz"
    save_volume(a, synthetic_affine, p1)
    save_volume(b, synthetic_affine, p2)
    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_subject_volumes(p1, p2)


# ---------------------------------------------------------------------------
# T010 — matching volumes pass without exception
# ---------------------------------------------------------------------------


def test_validate_subject_volumes_ok(synthetic_subject, synthetic_affine):
    t1_path = synthetic_subject / "t1.nii.gz"
    flair_path = synthetic_subject / "flair.nii.gz"
    t1, flair, affine = validate_subject_volumes(t1_path, flair_path)
    assert t1.shape == flair.shape
    assert t1.shape == (24, 32, 28)


# ---------------------------------------------------------------------------
# T011 — build_voxel_dataframe schema + row count
# ---------------------------------------------------------------------------


def test_build_voxel_dataframe_schema(spark_session, synthetic_subject):
    t1_path = synthetic_subject / "t1.nii.gz"
    flair_path = synthetic_subject / "flair.nii.gz"
    df = build_voxel_dataframe(
        spark_session, "subj_synth", str(t1_path), str(flair_path), partition_count=2
    )
    cols = df.columns
    for c in ["subject_id", "x", "y", "z", "t1", "flair"]:
        assert c in cols, f"Missing column: {c}"
    assert "label" not in cols
    assert df.count() == 24 * 32 * 28


# ---------------------------------------------------------------------------
# T015 — non-binary mask raises ValueError
# ---------------------------------------------------------------------------


def test_validate_binary_mask_fails_non_binary(tmp_path, synthetic_affine):
    arr = np.array([[[0.0, 1.0, 2.0]]], dtype=np.float32)
    p = tmp_path / "bad_mask.nii.gz"
    save_volume(arr, synthetic_affine, p)
    with pytest.raises(ValueError, match="non-binary"):
        validate_binary_mask(p)


# ---------------------------------------------------------------------------
# T016 — build_voxel_dataframe with mask appends label column
# ---------------------------------------------------------------------------


def test_build_voxel_dataframe_with_mask(spark_session, synthetic_subject):
    t1_path = synthetic_subject / "t1.nii.gz"
    flair_path = synthetic_subject / "flair.nii.gz"
    mask_path = synthetic_subject / "wmh_mask.nii.gz"
    df = build_voxel_dataframe(
        spark_session,
        "subj_synth",
        str(t1_path),
        str(flair_path),
        partition_count=2,
        mask_path=str(mask_path),
    )
    assert "label" in df.columns
    assert df.filter("label NOT IN (0, 1)").count() == 0
    assert df.filter("label = 1").count() > 0


# ---------------------------------------------------------------------------
# T017 — build_voxel_dataframe without mask has no label column
# ---------------------------------------------------------------------------


def test_build_voxel_dataframe_without_mask(spark_session, synthetic_subject):
    t1_path = synthetic_subject / "t1.nii.gz"
    flair_path = synthetic_subject / "flair.nii.gz"
    df = build_voxel_dataframe(
        spark_session,
        "subj_synth",
        str(t1_path),
        str(flair_path),
        partition_count=2,
        mask_path=None,
    )
    assert "label" not in df.columns


# ---------------------------------------------------------------------------
# T020 — derive_partition_count auto mode respects memory ceiling
# ---------------------------------------------------------------------------


def test_derive_partition_count_auto():
    result = derive_partition_count(voxel_count=500_000, bytes_per_row=28)
    assert isinstance(result, int)
    assert result >= 1
    assert 500_000 * 28 / result <= 16 * 1024**3


# ---------------------------------------------------------------------------
# T021 — derive_partition_count override returns exact value
# ---------------------------------------------------------------------------


def test_derive_partition_count_override():
    result = derive_partition_count(voxel_count=500_000, bytes_per_row=28, override=8)
    assert result == 8


# ---------------------------------------------------------------------------
# T022 — build_voxel_dataframe respects explicit and auto partition counts
# ---------------------------------------------------------------------------


def test_build_voxel_dataframe_partition_count(spark_session, synthetic_subject):
    t1_path = synthetic_subject / "t1.nii.gz"
    flair_path = synthetic_subject / "flair.nii.gz"

    # (a) explicit partition count
    df4 = build_voxel_dataframe(
        spark_session, "s", str(t1_path), str(flair_path), partition_count=4
    )
    assert df4.rdd.getNumPartitions() == 4

    # (b) partition_count=0 triggers auto-derive — must produce >= 1 partition
    df_auto = build_voxel_dataframe(
        spark_session, "s", str(t1_path), str(flair_path), partition_count=0
    )
    assert df_auto.rdd.getNumPartitions() >= 1


# ---------------------------------------------------------------------------
# T026a — sha256_of_file is stable and returns a 64-char hex string
# ---------------------------------------------------------------------------


def test_sha256_stable(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello wmh-spark")
    h1 = sha256_of_file(p)
    h2 = sha256_of_file(p)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


# ---------------------------------------------------------------------------
# T033 — missing T1 raises FileNotFoundError with path in message
# ---------------------------------------------------------------------------


def test_validate_subject_volumes_missing_t1(tmp_path, synthetic_subject, synthetic_affine):
    flair_path = synthetic_subject / "flair.nii.gz"
    missing = tmp_path / "nonexistent_t1.nii.gz"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        validate_subject_volumes(missing, flair_path)


# ---------------------------------------------------------------------------
# T034 — corrupt NIfTI raises IOError with file path in message
# ---------------------------------------------------------------------------


def test_load_volume_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.nii.gz"
    corrupt.write_bytes(b"\x00" * 10)
    with pytest.raises((IOError, OSError)):
        load_volume(corrupt)


# ---------------------------------------------------------------------------
# Extra: SubjectRecord round-trip
# ---------------------------------------------------------------------------


def test_subject_record_dict_roundtrip():
    rec = SubjectRecord(
        subject_id="s1",
        t1_path="/x/t1.nii.gz",
        flair_path="/x/flair.nii.gz",
    )
    d = rec.to_dict()
    assert d["subject_id"] == "s1"
    assert d["gt_mask_path"] is None
