"""End-to-end inference test: features -> trained model -> mask reconstruction.

Exercises the round trip without spinning up Spark. We use the k-NN baseline
because it has minimal overhead (no Spark MLlib import) and the same predict
contract as the other classifiers.
"""

from __future__ import annotations

import numpy as np

from wmh_spark.config import FeatureConfig, ModelConfig
from wmh_spark.features import extract_features
from wmh_spark.inference import infer_subject
from wmh_spark.io_utils import load_mask
from wmh_spark.models.knn_baseline import predict_knn, train_knn


def test_inference_roundtrip(tmp_path, synthetic_subject):
    # Extract features with ground truth so we can train.
    bundle = extract_features(
        subject_id="synth",
        flair_mni_path=str(synthetic_subject / "flair.nii.gz"),
        t1_mni_path=str(synthetic_subject / "t1.nii.gz"),
        brain_mask_mni_path=str(synthetic_subject / "brain_mask.nii.gz"),
        cfg=FeatureConfig(),
        gt_mask_mni_path=str(synthetic_subject / "wmh_mask.nii.gz"),
    )
    assert bundle.success

    # Train k-NN on the same subject (just a smoke test of the contract,
    # not a meaningful generalization claim).
    model = train_knn(bundle.features, bundle.labels, ModelConfig(name="knn", knn_k=3))

    out_dir = tmp_path / "out"
    result = infer_subject(
        bundle=bundle,
        predict_fn=lambda X: predict_knn(model, X),
        threshold=0.5,
        out_dir=str(out_dir),
    )
    assert result.success
    assert result.n_voxels == bundle.features.shape[0]

    # Reconstructed mask should be well-formed, in original volume shape.
    mask = load_mask(result.mask_path)
    assert mask.shape == bundle.volume_shape
    assert mask.dtype == np.uint8
    assert mask.sum() == result.n_predicted_lesion
