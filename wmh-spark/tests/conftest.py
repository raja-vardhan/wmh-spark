"""Shared test fixtures.

We synthesize tiny brain-shaped volumes so unit tests run in seconds
without needing real MRI data or a Spark cluster. The volumes are big
enough to exercise the pipeline logic and small enough that the entire
test suite finishes under a minute.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from wmh_spark.io_utils import save_volume


@pytest.fixture(scope="session")
def synthetic_volume_shape():
    return (24, 32, 28)


@pytest.fixture(scope="session")
def synthetic_affine():
    # 2mm isotropic, MNI-like origin.
    a = np.eye(4, dtype=np.float32) * 2
    a[3, 3] = 1.0
    a[:3, 3] = [-24.0, -32.0, -28.0]
    return a


def _make_brain(shape):
    """Make a roughly ellipsoidal 'brain' volume with known WM/CSF/lesion structure."""
    z, y, x = np.indices(shape, dtype=np.float32)
    cz, cy, cx = (s / 2 for s in shape)
    rz, ry, rx = (s / 2.5 for s in shape)
    ellipsoid = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    brain = ellipsoid < 1.0

    # Ventricles in the center.
    ventricles = ellipsoid < 0.05
    # WM is the brain minus the central ventricles.
    wm = brain & ~ventricles

    # Synth FLAIR: WM bright, ventricles dark, plus a small lesion.
    flair = np.zeros(shape, dtype=np.float32)
    flair[wm] = 100 + np.random.default_rng(0).normal(0, 5, wm.sum())
    flair[ventricles] = 20

    # Inject a hyperintense lesion.
    lesion = np.zeros(shape, dtype=bool)
    lz, ly, lx = (s // 2 - 3 for s in shape)
    lesion[lz:lz + 3, ly:ly + 4, lx:lx + 5] = True
    lesion &= wm
    flair[lesion] = 200

    # Synth T1: WM bright, ventricles dark. We keep the lesion T1 slightly
    # above the WM median so the WM-proxy quantile filter (used in
    # features.py to approximate a WM atlas) still includes lesion voxels
    # as candidates. Real deployments use an atlas-based WM mask and don't
    # have this constraint.
    t1 = np.zeros(shape, dtype=np.float32)
    t1[wm] = 150 + np.random.default_rng(1).normal(0, 5, wm.sum())
    t1[ventricles] = 30
    t1[lesion] = 152

    return flair, t1, brain.astype(np.uint8), lesion.astype(np.uint8)


@pytest.fixture
def synthetic_subject(tmp_path, synthetic_volume_shape, synthetic_affine):
    """One synthetic subject written to a tmp directory."""
    flair, t1, brain, lesion = _make_brain(synthetic_volume_shape)
    subj = tmp_path / "subj_synth"
    subj.mkdir()
    save_volume(flair, synthetic_affine, subj / "flair.nii.gz")
    save_volume(t1, synthetic_affine, subj / "t1.nii.gz")
    save_volume(brain, synthetic_affine, subj / "brain_mask.nii.gz", dtype=np.uint8)
    save_volume(lesion, synthetic_affine, subj / "wmh_mask.nii.gz", dtype=np.uint8)
    return subj
