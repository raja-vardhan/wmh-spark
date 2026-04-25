"""Self-contained validation harness.

In environments without nibabel/pyspark/pytest available, this script
exercises the pure-numpy/sklearn pieces of the codebase (features,
sampling, evaluation, classifiers) using an in-memory stub for nibabel.

This is *not* a substitute for the real test suite -- it's a smoke
test that the algorithmic logic is correct and consistent across modules.
Running on a real CHPC node, you would instead run ``pytest tests/``.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Stub nibabel using an in-memory dict so we don't need the real package.
# ---------------------------------------------------------------------------


_VOLUMES: dict[str, tuple[np.ndarray, np.ndarray]] = {}


class _StubImg:
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        self._data = data
        self.affine = affine

    @property
    def dataobj(self):
        return self._data


def _stub_load(path):
    key = str(path)
    if key not in _VOLUMES:
        raise FileNotFoundError(key)
    data, affine = _VOLUMES[key]
    return _StubImg(data, affine)


def _stub_save(img, path):
    _VOLUMES[str(path)] = (np.asarray(img._data), np.asarray(img.affine))


class _StubNifti1Image(_StubImg):
    pass


nibabel_stub = types.ModuleType("nibabel")
nibabel_stub.load = _stub_load
nibabel_stub.save = _stub_save
nibabel_stub.Nifti1Image = _StubNifti1Image
sys.modules["nibabel"] = nibabel_stub


# Now we can import the project modules.
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))

from wmh_spark.config import (  # noqa: E402
    FeatureConfig, ModelConfig, SamplingConfig, load_config,
)
from wmh_spark.evaluation import _dice, _hausdorff_95, _lesion_f1  # noqa: E402
from wmh_spark.features import FEATURE_NAMES, extract_features  # noqa: E402
from wmh_spark.inference import infer_subject  # noqa: E402
from wmh_spark.io_utils import load_mask, save_volume  # noqa: E402
from wmh_spark.models.knn_baseline import predict_knn, train_knn  # noqa: E402
from wmh_spark.sampling import sample_subject  # noqa: E402


# ---------------------------------------------------------------------------
# Tiny test runner
# ---------------------------------------------------------------------------


PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name: str, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILED.append((name, str(e) or "AssertionError"))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        FAILED.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


def approx(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} != {b} (tol={tol})"


# ---------------------------------------------------------------------------
# Synthetic subject generator (mirrors tests/conftest.py)
# ---------------------------------------------------------------------------


def make_synth_subject():
    shape = (24, 32, 28)
    affine = np.eye(4, dtype=np.float32) * 2
    affine[3, 3] = 1.0
    affine[:3, 3] = [-24, -32, -28]

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
    lesion[lz:lz + 3, ly:ly + 4, lx:lx + 5] = True
    lesion &= wm
    flair[lesion] = 200

    t1 = np.zeros(shape, dtype=np.float32)
    rng2 = np.random.default_rng(1)
    t1[wm] = 150 + rng2.normal(0, 5, wm.sum())
    t1[ventricles] = 30
    # Lesion T1 sits just above WM median so the quantile filter
    # (a stand-in for an atlas-based WM mask) still includes them.
    t1[lesion] = 152

    base = "/tmp/synth"
    save_volume(flair, affine, f"{base}/flair.nii.gz")
    save_volume(t1, affine, f"{base}/t1.nii.gz")
    save_volume(brain.astype(np.uint8), affine, f"{base}/brain_mask.nii.gz", dtype=np.uint8)
    save_volume(lesion.astype(np.uint8), affine, f"{base}/wmh_mask.nii.gz", dtype=np.uint8)
    return base, lesion.sum()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_io_roundtrip():
    affine = np.eye(4, dtype=np.float32)
    arr = np.random.default_rng(0).normal(size=(8, 9, 10)).astype(np.float32)
    save_volume(arr, affine, "/tmp/v.nii.gz")
    from wmh_spark.io_utils import load_volume
    loaded, aff = load_volume("/tmp/v.nii.gz")
    np.testing.assert_allclose(loaded, arr, atol=1e-5)
    np.testing.assert_allclose(aff, affine)


def test_features_runs():
    base, _ = make_synth_subject()
    bundle = extract_features(
        subject_id="synth",
        flair_mni_path=f"{base}/flair.nii.gz",
        t1_mni_path=f"{base}/t1.nii.gz",
        brain_mask_mni_path=f"{base}/brain_mask.nii.gz",
        cfg=FeatureConfig(),
        gt_mask_mni_path=f"{base}/wmh_mask.nii.gz",
    )
    assert bundle.success, f"feature extraction failed: {bundle.error}"
    assert bundle.features.shape[1] == len(FEATURE_NAMES)
    assert bundle.features.shape[0] > 0
    assert bundle.features.shape[0] == bundle.voxel_indices.shape[0]
    assert bundle.labels is not None
    assert bundle.labels.shape == (bundle.features.shape[0],)
    assert np.all(np.isfinite(bundle.features)), "non-finite features"


def test_lesion_separates_in_flair_z():
    base, _ = make_synth_subject()
    bundle = extract_features(
        "synth",
        f"{base}/flair.nii.gz", f"{base}/t1.nii.gz",
        f"{base}/brain_mask.nii.gz",
        FeatureConfig(),
        f"{base}/wmh_mask.nii.gz",
    )
    flair_z = bundle.features[:, 0]
    pos_mean = flair_z[bundle.labels == 1].mean()
    neg_mean = flair_z[bundle.labels == 0].mean()
    assert pos_mean > neg_mean, f"pos={pos_mean} should exceed neg={neg_mean}"


def test_dice_metrics():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[3:7, 3:7, 3:7] = True
    approx(_dice(a, a), 1.0)
    b = np.zeros_like(a)
    b[7:, 7:, 7:] = True
    approx(_dice(a, b), 0.0)
    z = np.zeros_like(a)
    approx(_dice(z, z), 1.0)


def test_lesion_f1_one_to_one():
    a = np.zeros((20, 20, 20), dtype=bool)
    b = np.zeros_like(a)
    a[2:5, 2:5, 2:5] = True
    a[10:13, 10:13, 10:13] = True
    b[2:6, 2:6, 2:6] = True
    b[10:14, 10:14, 10:14] = True
    p, r, f1 = _lesion_f1(b, a)
    approx(p, 1.0)
    approx(r, 1.0)
    approx(f1, 1.0)


def test_lesion_f1_extra_fp():
    ref = np.zeros((20, 20, 20), dtype=bool)
    pred = np.zeros_like(ref)
    ref[2:5, 2:5, 2:5] = True
    pred[2:5, 2:5, 2:5] = True
    pred[15:18, 15:18, 15:18] = True
    p, r, f1 = _lesion_f1(pred, ref)
    approx(r, 1.0)
    approx(p, 0.5)


def test_hausdorff():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[3:7, 3:7, 3:7] = True
    approx(_hausdorff_95(a, a), 0.0)
    z = np.zeros((10, 10, 10), dtype=bool)
    assert np.isnan(_hausdorff_95(z, z))


def test_sampling_ratio_respected():
    n_pos, n_neg = 100, 10_000
    feats = np.random.default_rng(0).normal(size=(n_pos + n_neg, len(FEATURE_NAMES))).astype(np.float32)
    labels = np.concatenate([np.ones(n_pos, dtype=np.uint8), np.zeros(n_neg, dtype=np.uint8)])
    from wmh_spark.features import FeatureBundle
    bundle = FeatureBundle(
        subject_id="s",
        features=feats,
        voxel_indices=np.arange(n_pos + n_neg, dtype=np.int64),
        volume_shape=(1, 1, n_pos + n_neg),
        affine=np.eye(4),
        labels=labels,
        success=True,
    )
    cfg = SamplingConfig(negative_to_positive_ratio=5, max_voxels_per_subject=10_000)
    X, y = sample_subject(bundle, cfg, rng_seed=42)
    n_p = int((y == 1).sum())
    n_n = int((y == 0).sum())
    assert n_p > 0
    assert n_n <= cfg.negative_to_positive_ratio * n_p, f"{n_n} exceeds 5x{n_p}"


def test_sampling_deterministic():
    feats = np.random.default_rng(0).normal(size=(1000, len(FEATURE_NAMES))).astype(np.float32)
    labels = np.zeros(1000, dtype=np.uint8)
    labels[:50] = 1
    from wmh_spark.features import FeatureBundle
    bundle = FeatureBundle(
        subject_id="s", features=feats,
        voxel_indices=np.arange(1000, dtype=np.int64),
        volume_shape=(1, 1, 1000), affine=np.eye(4),
        labels=labels, success=True,
    )
    cfg = SamplingConfig()
    X1, y1 = sample_subject(bundle, cfg, rng_seed=42)
    X2, y2 = sample_subject(bundle, cfg, rng_seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_inference_roundtrip():
    base, lesion_size = make_synth_subject()
    bundle = extract_features(
        "synth",
        f"{base}/flair.nii.gz", f"{base}/t1.nii.gz",
        f"{base}/brain_mask.nii.gz",
        FeatureConfig(),
        f"{base}/wmh_mask.nii.gz",
    )
    model = train_knn(bundle.features, bundle.labels, ModelConfig(name="knn", knn_k=3))
    result = infer_subject(
        bundle=bundle,
        predict_fn=lambda X: predict_knn(model, X),
        threshold=0.5,
        out_dir="/tmp/out",
    )
    assert result.success, f"inference failed: {result.error}"
    assert result.n_voxels == bundle.features.shape[0]
    mask = load_mask(result.mask_path)
    assert mask.shape == bundle.volume_shape


def test_config_loads():
    cfg_path = PROJECT / "configs" / "local.yaml"
    cfg = load_config(cfg_path)
    assert cfg.seed == 42
    assert cfg.spark.master.startswith("local")
    assert cfg.model.name in {"rf", "knn", "xgb"}


def test_partial_yaml_uses_defaults(tmp="/tmp/partial.yaml"):
    Path(tmp).write_text("seed: 7\n")
    cfg = load_config(tmp)
    assert cfg.seed == 7
    assert cfg.features.neighborhood_radius == 1


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("Running validation suite...")
    print()
    print("Config:")
    check("config_local_yaml_loads", test_config_loads)
    check("config_partial_yaml_uses_defaults", test_partial_yaml_uses_defaults)
    print()
    print("IO:")
    check("io_save_load_roundtrip", test_io_roundtrip)
    print()
    print("Features:")
    check("features_extract_runs", test_features_runs)
    check("features_lesion_separates_in_flair_z", test_lesion_separates_in_flair_z)
    print()
    print("Evaluation:")
    check("eval_dice", test_dice_metrics)
    check("eval_lesion_f1_one_to_one", test_lesion_f1_one_to_one)
    check("eval_lesion_f1_extra_fp", test_lesion_f1_extra_fp)
    check("eval_hausdorff", test_hausdorff)
    print()
    print("Sampling:")
    check("sampling_ratio_respected", test_sampling_ratio_respected)
    check("sampling_deterministic", test_sampling_deterministic)
    print()
    print("Inference:")
    check("inference_roundtrip", test_inference_roundtrip)
    print()
    print("=" * 60)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for name, msg in FAILED:
            print(f"  FAILED: {name}: {msg}")
        sys.exit(1)
