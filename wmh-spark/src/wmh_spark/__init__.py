"""wmh-spark public API."""

from wmh_spark.feature_extraction import (
    FeaturePartitionPlan,
    add_t1_flair_ratio,
    build_feature_dataframe,
    build_spatial_prior_dataframe,
    derive_feature_partition_count,
    validate_spatial_prior_template,
)
from wmh_spark.evaluation import (
    AccuracyGateError,
    DiceResult,
    EvaluationConfig,
    assert_reproduction_accuracy,
    calculate_dice_result,
    log_cluster_throughput,
    log_subject_benchmark,
    summarize_scaling,
)
from wmh_spark.models import (
    ClassificationConfig,
    predict_voxel_mask,
    prepare_training_dataframe,
    train_random_forest_model,
)
from wmh_spark.postprocessing import (
    PostProcessingConfig,
    component_size_map,
    label_connected_components,
    postprocess_predictions,
    reconstruct_prediction_volume,
)

__all__ = [
    "AccuracyGateError",
    "ClassificationConfig",
    "DiceResult",
    "EvaluationConfig",
    "FeaturePartitionPlan",
    "PostProcessingConfig",
    "add_t1_flair_ratio",
    "assert_reproduction_accuracy",
    "build_feature_dataframe",
    "build_spatial_prior_dataframe",
    "calculate_dice_result",
    "component_size_map",
    "derive_feature_partition_count",
    "label_connected_components",
    "log_cluster_throughput",
    "log_subject_benchmark",
    "predict_voxel_mask",
    "postprocess_predictions",
    "prepare_training_dataframe",
    "reconstruct_prediction_volume",
    "summarize_scaling",
    "train_random_forest_model",
    "validate_spatial_prior_template",
]
