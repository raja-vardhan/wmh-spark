"""Spark MLlib Random Forest classifier for voxel-level WMH prediction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
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
    positive_probability_column: str = "wmh_probability"
    raw_prediction_column: str = "raw_prediction"
    mask_column: str = "predicted_mask"
    weight_column: str = "class_weight"
    training_partitions: int = 4
    num_trees: int = 20
    max_depth: int = 8
    max_bins: int = 32
    seed: int = 42
    subsampling_rate: float = 1.0
    prediction_threshold: float = 0.25
    positive_class_weight_cap: float = 100.0


@dataclass(frozen=True)
class ClassBalanceStats:
    """Training label counts and the resolved WMH class weight."""

    negative_count: int
    positive_count: int
    positive_class_weight: float

    def to_dict(self) -> dict:
        return {
            "negative_count": self.negative_count,
            "positive_count": self.positive_count,
            "positive_class_weight": self.positive_class_weight,
        }


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


def _validate_probability_threshold(config: ClassificationConfig) -> None:
    if not 0.0 <= config.prediction_threshold <= 1.0:
        raise ValueError("prediction_threshold must be between 0 and 1")


def _validate_weight_cap(config: ClassificationConfig) -> None:
    if config.positive_class_weight_cap <= 0:
        raise ValueError("positive_class_weight_cap must be positive")


def _assembler(config: ClassificationConfig) -> VectorAssembler:
    return VectorAssembler(
        inputCols=list(config.feature_columns),
        outputCol=config.features_column,
        handleInvalid="keep",
    )


def calculate_class_balance_stats(
    df: "DataFrame",
    config: ClassificationConfig,
) -> ClassBalanceStats:
    """Return binary label counts and capped positive-class weight."""
    validate_feature_columns(df, config, require_label=True)
    validate_binary_labels(df, config)
    _validate_weight_cap(config)

    counts = {
        int(row[config.label_column]): int(row["count"])
        for row in df.groupBy(config.label_column).count().collect()
    }
    negative_count = counts.get(0, 0)
    positive_count = counts.get(1, 0)
    if negative_count == 0 or positive_count == 0:
        raise ValueError(
            "training labels must contain both background and WMH classes "
            f"(negative_count={negative_count}, positive_count={positive_count})"
        )

    positive_weight = min(
        negative_count / positive_count,
        float(config.positive_class_weight_cap),
    )
    return ClassBalanceStats(
        negative_count=negative_count,
        positive_count=positive_count,
        positive_class_weight=float(positive_weight),
    )


def add_class_weight_column(
    df: "DataFrame",
    config: ClassificationConfig,
    stats: ClassBalanceStats | None = None,
) -> "DataFrame":
    """Append the configured RF weight column for imbalanced WMH training."""
    stats = stats or calculate_class_balance_stats(df, config)
    return df.withColumn(
        config.weight_column,
        F.when(
            F.col(config.label_column).cast("int") == F.lit(1),
            F.lit(float(stats.positive_class_weight)),
        ).otherwise(F.lit(1.0)),
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

    prepared = _assembler(config).transform(add_class_weight_column(df, config))
    return (
        prepared
        # Repartition to the classifier plan so Spark ML tree training has work
        # available across the four-worker CHPC target and bounded shuffle chunks.
        .repartition(config.training_partitions)
    )


def train_random_forest_model(
    df: "DataFrame",
    config: ClassificationConfig | None = None,
    balance_stats: ClassBalanceStats | None = None,
) -> "PipelineModel":
    """Train a distributed Spark MLlib Random Forest model on Feature 3 rows."""
    config = config or ClassificationConfig()
    _validate_training_partitions(config)
    _validate_probability_threshold(config)
    validate_feature_columns(df, config, require_label=True)
    validate_binary_labels(df, config)
    balance_stats = balance_stats or calculate_class_balance_stats(df, config)

    spark = df.sparkSession
    spark.conf.set("spark.sql.shuffle.partitions", str(config.training_partitions))
    training_df = (
        add_class_weight_column(df, config, balance_stats)
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
        weightCol=config.weight_column,
    )
    pipeline = Pipeline(stages=[_assembler(config), classifier])
    logger.info(
        (
            "training RandomForestClassifier: partitions=%d trees=%d max_depth=%d "
            "negative_count=%d positive_count=%d positive_weight=%.4f"
        ),
        training_df.rdd.getNumPartitions(),
        config.num_trees,
        config.max_depth,
        balance_stats.negative_count,
        balance_stats.positive_count,
        balance_stats.positive_class_weight,
    )
    return pipeline.fit(training_df)


def predict_voxel_mask(
    model: "PipelineModel",
    df: "DataFrame",
    config: ClassificationConfig | None = None,
) -> "DataFrame":
    """Apply a trained model and append an integer raw binary voxel mask column."""
    config = config or ClassificationConfig()
    _validate_probability_threshold(config)
    validate_feature_columns(df, config, require_label=False)

    predicted = model.transform(df)
    positive_probability = vector_to_array(F.col(config.probability_column))[1]
    return (
        predicted.withColumn(
            config.positive_probability_column,
            positive_probability.cast("double"),
        )
        .withColumn(
            config.mask_column,
            (
                F.col(config.positive_probability_column)
                >= F.lit(float(config.prediction_threshold))
            ).cast("int"),
        )
    )
