"""Tests for io_utils."""

from __future__ import annotations

import numpy as np

from wmh_spark.io_utils import (
    SubjectRecord,
    load_mask,
    load_volume,
    save_volume,
    sha256_of_file,
)


def test_save_load_roundtrip(tmp_path, synthetic_affine):
    arr = np.random.default_rng(0).normal(size=(8, 9, 10)).astype(np.float32)
    out = tmp_path / "vol.nii.gz"
    save_volume(arr, synthetic_affine, out)
    loaded, affine = load_volume(out)
    np.testing.assert_allclose(loaded, arr, atol=1e-5)
    np.testing.assert_allclose(affine, synthetic_affine)


def test_load_mask_binarizes(tmp_path, synthetic_affine):
    arr = np.array([[[0.0, 0.5, 1.0], [2.0, 0.0, -1.0]]], dtype=np.float32)
    out = tmp_path / "m.nii.gz"
    save_volume(arr, synthetic_affine, out)
    mask = load_mask(out)
    assert mask.dtype == np.uint8
    # Anything > 0 is foreground; negative and zero are background.
    expected = np.array([[[0, 1, 1], [1, 0, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(mask, expected)


def test_subject_record_dict_roundtrip():
    rec = SubjectRecord(
        subject_id="s1",
        flair_path="/x/flair.nii.gz",
        t1_path="/x/t1.nii.gz",
        gt_mask_path=None,
        site="A",
        age=72.0,
        split="train",
    )
    d = rec.to_dict()
    assert d["subject_id"] == "s1"
    assert d["age"] == 72.0
    assert d["gt_mask_path"] is None


def test_sha256_stable(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello world")
    h1 = sha256_of_file(p)
    h2 = sha256_of_file(p)
    assert h1 == h2
    assert len(h1) == 64
