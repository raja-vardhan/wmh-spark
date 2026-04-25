"""End-to-end pipeline orchestration.

This is the only file that holds Spark idioms. It assembles the stages
defined in the other modules into a single Spark application.

Architectural reminders:
- ``rdd.map`` over subjects for I/O-bound stages (strip, register, features,
  inference). Each record is one subject; the worker does the work locally.
- A Spark DataFrame is used only for the pooled training set, where
  distributed shuffling actually pays off.
- The trained model is broadcast back to workers for inference -- shipping
  a model object is cheap; shipping every subject's volume is not.

The pipeline is checkpointed at the end of each stage. Re-running picks up
where the last successful stage finished.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from .config import Config, dump_config
from .features import extract_features
from .inference import infer_subject
from .registration import register_subject
from .stripping import strip_subject

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------


def build_spark(cfg: Config):
    """Construct a SparkSession from cfg.spark.

    We pin Arrow batch size so Pandas UDFs (used in the sampling stage) don't
    spill memory on workers when ndarrays travel through them.
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .appName(cfg.spark.app_name)
        .master(cfg.spark.master)
        .config("spark.driver.memory", cfg.spark.driver_memory)
        .config("spark.executor.memory", cfg.spark.executor_memory)
        .config("spark.executor.cores", str(cfg.spark.executor_cores))
        .config("spark.executor.instances", str(cfg.spark.num_executors))
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config(
            "spark.sql.execution.arrow.maxRecordsPerBatch",
            str(cfg.spark.arrow_max_records_per_batch),
        )
        .config("spark.pyspark.python", cfg.spark.python_exec)
    )
    for key, val in cfg.spark.extra_conf.items():
        builder = builder.config(key, val)

    return builder.getOrCreate()


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------


def _stage_strip(spark, manifest_df, cfg: Config):
    """Map skull stripping over the subject RDD."""
    rows = manifest_df.select(
        "subject_id", "flair_path", "t1_path"
    ).rdd.map(tuple).collect()

    rdd = spark.sparkContext.parallelize(rows, numSlices=max(len(rows), 1))
    work_dir = f"{cfg.paths.work_dir}/01_strip"
    strip_cfg = cfg.stripping
    results = rdd.map(
        lambda r: strip_subject(r[0], r[1], r[2], work_dir, strip_cfg)
    ).collect()
    return results


def _stage_register(spark, strip_results, cfg: Config):
    successful = [r for r in strip_results if r.success]
    rows = [(r.subject_id, r.flair_brain_path, r.t1_brain_path, r.brain_mask_path)
            for r in successful]
    rdd = spark.sparkContext.parallelize(rows, numSlices=max(len(rows), 1))

    work_dir = f"{cfg.paths.work_dir}/02_register"
    reg_cfg = cfg.registration
    results = rdd.map(
        lambda r: register_subject(r[0], r[1], r[2], r[3], work_dir, reg_cfg)
    ).collect()
    return results


def _stage_features(spark, reg_results, manifest_df, cfg: Config):
    """Extract features per subject. Joins ground-truth paths from manifest."""
    successful = [r for r in reg_results if r.success]
    gt_lookup = {
        row["subject_id"]: row["gt_mask_path"]
        for row in manifest_df.collect()
    }
    rows = [
        (r.subject_id, r.flair_mni_path, r.t1_mni_path,
         r.brain_mask_mni_path, gt_lookup.get(r.subject_id))
        for r in successful
    ]
    rdd = spark.sparkContext.parallelize(rows, numSlices=max(len(rows), 1))
    feat_cfg = cfg.features
    bundles = rdd.map(
        lambda r: extract_features(r[0], r[1], r[2], r[3], feat_cfg, r[4])
    ).collect()
    return bundles


def _stage_train(spark, feature_bundles, cfg: Config):
    """Pool subsampled voxels into a Spark DataFrame and train the model."""
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import Row

    from .features import FEATURE_NAMES
    from .sampling import sample_subject

    successful = [b for b in feature_bundles if b.success and b.labels is not None]
    if not successful:
        raise RuntimeError("No labeled subjects available for training")

    # Sample on the driver (cheap; per-subject features already fit in RAM).
    sampled: list[Row] = []
    for b in successful:
        seed = (cfg.seed + hash(b.subject_id)) & 0xFFFFFFFF
        X, y = sample_subject(b, cfg.sampling, seed)
        for row, label in zip(X, y):
            sampled.append(Row(features=Vectors.dense(row.tolist()), label=int(label)))

    if not sampled:
        raise RuntimeError("Sampling produced no rows")

    train_df = spark.createDataFrame(sampled)
    n_pos = sum(1 for r in sampled if r.label == 1)
    logger.info("Pooled training set: %d rows (%d positive)", len(sampled), n_pos)

    # Dispatch to the configured classifier.
    name = cfg.model.name
    if name == "rf":
        from .models.rf_spark import train_rf_spark
        bundle = train_rf_spark(spark, train_df, cfg.model, cfg.paths.model_dir)
        predict_fn_factory = _rf_predict_factory(bundle)
    elif name == "knn":
        from .models.knn_baseline import train_knn
        import numpy as np
        X = np.array([r.features.toArray() for r in sampled], dtype=np.float32)
        y = np.array([r.label for r in sampled], dtype=np.uint8)
        bundle = train_knn(X, y, cfg.model)
        predict_fn_factory = _knn_predict_factory(bundle)
    elif name == "xgb":
        from .models.xgb_spark import train_xgb_spark
        import numpy as np
        X = np.array([r.features.toArray() for r in sampled], dtype=np.float32)
        y = np.array([r.label for r in sampled], dtype=np.uint8)
        bundle = train_xgb_spark(X, y, cfg.model)
        predict_fn_factory = _xgb_predict_factory(bundle)
    else:
        raise ValueError(f"Unknown model: {name}")

    return bundle, predict_fn_factory


def _rf_predict_factory(bundle):
    from .models.rf_spark import predict_rf_spark
    def factory():
        return lambda X: predict_rf_spark(bundle, X)
    return factory


def _knn_predict_factory(bundle):
    from .models.knn_baseline import predict_knn
    def factory():
        return lambda X: predict_knn(bundle, X)
    return factory


def _xgb_predict_factory(bundle):
    from .models.xgb_spark import predict_xgb_spark
    def factory():
        return lambda X: predict_xgb_spark(bundle, X)
    return factory


def _stage_infer(spark, feature_bundles, predict_fn_factory, cfg: Config):
    """Broadcast the model and run per-subject inference."""
    successful = [b for b in feature_bundles if b.success]
    bcast = spark.sparkContext.broadcast(predict_fn_factory)

    out_dir = cfg.paths.output_dir
    threshold = cfg.evaluation.decision_threshold
    rdd = spark.sparkContext.parallelize(successful, numSlices=max(len(successful), 1))

    def _runner(bundle):
        predict_fn = bcast.value()
        return infer_subject(bundle, predict_fn, threshold, out_dir)

    return rdd.map(_runner).collect()


def _stage_evaluate(spark, infer_results, manifest_df, cfg: Config):
    """Compute metrics for every subject that has a reference mask."""
    from .evaluation import evaluate_subject

    gt_lookup = {row["subject_id"]: row["gt_mask_path"] for row in manifest_df.collect()}
    ubo_lookup = {row["subject_id"]: row["ubo_mask_path"] for row in manifest_df.collect()}

    metrics = []
    voxel_volume_ml = 1e-3  # 1mm isotropic MNI; adjust if template differs
    eval_cfg = cfg.evaluation

    for r in infer_results:
        if not r.success:
            continue
        gt = gt_lookup.get(r.subject_id)
        if gt:
            metrics.append(asdict(_eval_safe(
                r.subject_id, r.mask_path, gt, "expert", voxel_volume_ml, eval_cfg
            )))
        ubo = ubo_lookup.get(r.subject_id)
        if ubo:
            metrics.append(asdict(_eval_safe(
                r.subject_id, r.mask_path, ubo, "ubo", voxel_volume_ml, eval_cfg
            )))

    metrics_path = Path(cfg.paths.metrics_dir) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def _eval_safe(subject_id, pred_path, ref_path, ref_name, voxel_volume_ml, cfg):
    from .evaluation import evaluate_subject, SubjectMetrics
    try:
        return evaluate_subject(subject_id, pred_path, ref_path, ref_name, voxel_volume_ml, cfg)
    except Exception as exc:
        logger.exception("Evaluation failed for %s vs %s", subject_id, ref_name)
        return SubjectMetrics(
            subject_id=subject_id, reference=ref_name,
            dice=float("nan"), lesion_f1=float("nan"),
            lesion_precision=float("nan"), lesion_recall=float("nan"),
            hd95_mm=float("nan"), abs_vol_diff_ml=float("nan"),
            pred_volume_ml=float("nan"), ref_volume_ml=float("nan"),
        )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_pipeline(cfg: Config) -> dict:
    """Execute every stage end to end. Returns a summary dict for logging."""
    Path(cfg.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.model_dir).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg.paths.output_dir) / "resolved_config.yaml")

    spark = build_spark(cfg)
    spark.sparkContext.setLogLevel("WARN")

    summary: dict = {"timings": {}, "counts": {}}
    t_total = time.perf_counter()

    manifest_df = spark.read.parquet(cfg.paths.manifest_path)
    summary["counts"]["subjects"] = manifest_df.count()
    logger.info("Loaded manifest: %d subjects", summary["counts"]["subjects"])

    t = time.perf_counter()
    strip_results = _stage_strip(spark, manifest_df, cfg)
    summary["timings"]["strip_s"] = time.perf_counter() - t
    summary["counts"]["strip_ok"] = sum(1 for r in strip_results if r.success)

    t = time.perf_counter()
    reg_results = _stage_register(spark, strip_results, cfg)
    summary["timings"]["register_s"] = time.perf_counter() - t
    summary["counts"]["register_ok"] = sum(1 for r in reg_results if r.success)

    t = time.perf_counter()
    feature_bundles = _stage_features(spark, reg_results, manifest_df, cfg)
    summary["timings"]["features_s"] = time.perf_counter() - t
    summary["counts"]["features_ok"] = sum(1 for b in feature_bundles if b.success)

    t = time.perf_counter()
    model_bundle, predict_fn_factory = _stage_train(spark, feature_bundles, cfg)
    summary["timings"]["train_s"] = time.perf_counter() - t

    t = time.perf_counter()
    infer_results = _stage_infer(spark, feature_bundles, predict_fn_factory, cfg)
    summary["timings"]["infer_s"] = time.perf_counter() - t
    summary["counts"]["infer_ok"] = sum(1 for r in infer_results if r.success)

    t = time.perf_counter()
    metrics = _stage_evaluate(spark, infer_results, manifest_df, cfg)
    summary["timings"]["evaluate_s"] = time.perf_counter() - t
    summary["counts"]["metrics_records"] = len(metrics)

    summary["timings"]["total_s"] = time.perf_counter() - t_total
    with open(Path(cfg.paths.output_dir) / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    spark.stop()
    return summary
