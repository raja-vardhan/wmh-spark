"""Tests for evaluation metrics.

Pure-numpy tests on synthesized masks; no Spark or filesystem dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

from wmh_spark.evaluation import _dice, _hausdorff_95, _lesion_f1


def test_dice_perfect_overlap():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[3:7, 3:7, 3:7] = True
    assert _dice(a, a) == pytest.approx(1.0)


def test_dice_no_overlap():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros_like(a)
    a[0:3, 0:3, 0:3] = True
    b[7:10, 7:10, 7:10] = True
    assert _dice(a, b) == pytest.approx(0.0)


def test_dice_partial():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros_like(a)
    a[0:5, :, :] = True
    b[3:8, :, :] = True
    # Intersection = 2*100 = 200 voxels; |a|=500, |b|=500 -> Dice=2*200/(500+500)=0.4
    assert _dice(a, b) == pytest.approx(0.4)


def test_dice_both_empty_is_one():
    z = np.zeros((4, 4, 4), dtype=bool)
    assert _dice(z, z) == pytest.approx(1.0)


def test_lesion_f1_one_to_one():
    a = np.zeros((20, 20, 20), dtype=bool)
    b = np.zeros_like(a)
    a[2:5, 2:5, 2:5] = True   # lesion 1
    a[10:13, 10:13, 10:13] = True  # lesion 2
    b[2:6, 2:6, 2:6] = True   # overlaps lesion 1 (slightly bigger)
    b[10:14, 10:14, 10:14] = True  # overlaps lesion 2

    p, r, f1 = _lesion_f1(b, a)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)


def test_lesion_f1_extra_false_positive():
    ref = np.zeros((20, 20, 20), dtype=bool)
    pred = np.zeros_like(ref)
    ref[2:5, 2:5, 2:5] = True
    pred[2:5, 2:5, 2:5] = True
    pred[15:18, 15:18, 15:18] = True   # extra FP lesion
    p, r, f1 = _lesion_f1(pred, ref)
    assert r == pytest.approx(1.0)
    assert p == pytest.approx(0.5)


def test_hausdorff_empty_returns_nan():
    z = np.zeros((10, 10, 10), dtype=bool)
    assert np.isnan(_hausdorff_95(z, z))


def test_hausdorff_identical_zero():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[3:7, 3:7, 3:7] = True
    assert _hausdorff_95(a, a) == pytest.approx(0.0)
