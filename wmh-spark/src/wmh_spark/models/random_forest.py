"""Spark MLlib Random Forest classifier for voxel-level WMH prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F

if TYPE_CHECKING:
    from pyspark.ml import PipelineModel
    from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLUMNS = ("t1", "flair", "t1_flair_ratio", "spatial_prior")


@dataclass(frozen=True)
class ClassificationConfig:
    """Reproducible settings for the Spark MLlib Random Forest classifier."""

    feature_columns: Sequence[str] = field(default_factory=lambda: DEFAULT_FEATURE_COLUMNS)
    label_column: str = "label"
    features_column: str = "features"
    prediction_column: str = "prediction"
    probability_column: str = "probability"
    raw_prediction_column: str = "raw_prediction"
    mask_column: str = "predicted_mask"
    training_partitions: int = 4
    num_trees: int = 20
    max_depth: int = 8
    max_bins: int = 32
    seed: int = 42
    subsampling_rate: float = 1.0


def _missing_columns(df: "DataFrame", required: Sequence[str]) -> list[str]:
    present = set(df.columns)
    return [column for column in required if column not in present]


def validate_feature_columns(
    df: "DataFrame",
    config: ClassificationConfig,
    require_label: bool = True,
) -> None:
    """Validate that a Feature 3 matrix has all columns required by the model."""
    required = list(config.feature_columns)
    if require_label:
        required.append(config.label_column)

    missing = _missing_columns(df, required)
    if missing:
        raise ValueError(f"missing required classification columns: {missing}")


def validate_binary_labels(df: "DataFrame", config: ClassificationConfig) -> None:
    """Fail fast if labels are not binary 0/1 values."""
    labels = {
        row[config.label_column]
        for row in df.select(config.label_column).distinct().limit(3).collect()
    }
    invalid = sorted(label for label in labels if label not in (0, 1, 0.0, 1.0))
    if invalid:
        raise ValueError(
            f"label column '{config.label_column}' must be binary 0/1; "
            f"found {invalid}"
        )


def _validate_training_partitions(config: ClassificationConfig) -> None:
    if config.training_partitions <= 0:
        raise ValueError("training_partitions must be positive")


def _assembler(config: ClassificationConfig) -> VectorAssembler:
    return VectorAssembler(
        inputCols=list(config.feature_columns),
        outputCol=config.features_column,
        handleInvalid="keep",
    )


def prepare_training_dataframe(
    df: "DataFrame",
    config: ClassificationConfig,
) -> "DataFrame":
    """Assemble feature vectors and apply the configured training partition plan."""
    _validate_training_partitions(config)
    validate_feature_columns(df, config, require_label=True)
    validate_binary_labels(df, config)

    spark = df.sparkSession
    spark.conf.set("spark.sql.shuffle.partitions", str(config.training_partitions))

    prepared = _assembler(config).transform(df)
    return (
        prepared
        # Repartition to the classifier plan so Spark ML tree training has work
        # available across the four-worker CHPC target and bounded shuffle chunks.
        .repartition(config.training_partitions)
    )


def train_random_forest_model(
    df: "DataFrame",
    config: ClassificationConfig | None = None,
) -> "PipelineModel":
    """Train a distributed Spark MLlib Random Forest model on Feature 3 rows."""
    config = config or ClassificationConfig()
    _validate_training_partitions(config)
    validate_feature_columns(df, config, require_label=True)
    validate_binary_labels(df, config)

    spark = df.sparkSession
    spark.conf.set("spark.sql.shuffle.partitions", str(config.training_partitions))
    training_df = (
        df
        # Repartition to the classifier plan so RandomForestClassifier can train
        # over distributed partitions instead of a collapsed local table.
        .repartition(config.training_partitions)
    )

    classifier = RandomForestClassifier(
        featuresCol=config.features_column,
        labelCol=config.label_column,
        predictionCol=config.prediction_column,
        probabilityCol=config.probability_column,
        rawPredictionCol=config.raw_prediction_column,
        numTrees=config.num_trees,
        maxDepth=config.max_depth,
        maxBins=config.max_bins,
        seed=config.seed,
        subsamplingRate=config.subsampling_rate,
    )
    pipeline = Pipeline(stages=[_assembler(config), classifier])
    logger.info(
        "training RandomForestClassifier: partitions=%d trees=%d max_depth=%d",
        training_df.rdd.getNumPartitions(),
        config.num_trees,
        config.max_depth,
    )
    return pipeline.fit(training_df)


def predict_voxel_mask(
    model: "PipelineModel",
    df: "DataFrame",
    config: ClassificationConfig | None = None,
) -> "DataFrame":
    """Apply a trained model and append an integer raw binary voxel mask column."""
    config = config or ClassificationConfig()
    validate_feature_columns(df, config, require_label=False)

    predicted = model.transform(df)
    return predicted.withColumn(
        config.mask_column,
        F.col(config.prediction_column).cast("int"),
    )
