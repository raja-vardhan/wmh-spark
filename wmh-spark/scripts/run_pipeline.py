"""End-to-end WMH pipeline driver.

Stages:
  1. arrange dataset -> per-split manifest.parquet
  2. HD-BET skull-strip (train + test)
  3. build per-subject spatial prior from FLAIR brain mask
  4. build training feature DataFrame, union across train subjects
  5. train Spark MLlib Random Forest, persist model
  6. predict + postprocess per test subject -> wmh_pred.nii.gz
  7. compute Dice + lesion stats per test subject
  8. render per-subject overlay + report.html
  9. write aggregate report.html, metrics.csv, plots
 10. write benchmark JSONL + summary.json

Run:
  wmh-spark/.venv/bin/python wmh-spark/scripts/run_pipeline.py \\
      --data-root datasets/kaggle/wmh_data \\
      --layout kaggle-wmh \\
      --output-root wmh-spark/data/output/runs/pilot \\
      --train-sites Amsterdam,Singapore --test-sites Amsterdam,Singapore \\
      --scanners GE3T \\
      --max-train 5 --max-test 5
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from collections import Counter
from dataclasses import dataclass
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, TypeVar

import numpy as np

# Keep PySpark workers on the same Python interpreter as the driver.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "wmh-spark" / "src"))

import nibabel as nib  # noqa: E402
from pyspark.sql import DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from tqdm import tqdm  # noqa: E402
from tqdm.contrib.logging import logging_redirect_tqdm  # noqa: E402

from wmh_spark.dataset_layout import (  # noqa: E402
    arrange_kaggle_wmh,
    arrange_generic,
    subset,
    write_manifest,
)
from wmh_spark.evaluation import (  # noqa: E402
    EvaluationConfig,
    calculate_dice_result,
    log_subject_benchmark,
)
from wmh_spark.feature_extraction import build_feature_dataframe  # noqa: E402
from wmh_spark.io_utils import SubjectRecord, save_volume  # noqa: E402
from wmh_spark.models import (  # noqa: E402
    ClassificationConfig,
    ClassBalanceStats,
    calculate_class_balance_stats,
    predict_voxel_mask,
    train_random_forest_model,
)
from wmh_spark.postprocessing import (  # noqa: E402
    PostProcessingConfig,
    postprocess_predictions,
)
from wmh_spark.preprocessing.skull_strip import SkullStripper  # noqa: E402
from wmh_spark.reporting import (  # noqa: E402
    SubjectMetrics,
    count_lesions,
    render_subject_overlay,
    voxel_volume_mm3,
    write_aggregate_report,
    write_subject_report,
)

logger = logging.getLogger("run_pipeline")
T = TypeVar("T")
PAIR_AFFINE_ATOL = 1e-4
GT_AFFINE_ATOL = 1e-3


@dataclass
class ArrangedDataset:
    train_discovered: List[SubjectRecord]
    test_discovered: List[SubjectRecord]
    train_selected: List[SubjectRecord]
    test_selected: List[SubjectRecord]
    manifests_dir: Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end WMH pipeline driver")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument(
        "--layout", choices=["kaggle-wmh", "generic"], default="kaggle-wmh"
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument(
        "--train-sites",
        type=str,
        default="Amsterdam,Singapore",
        help="Comma-separated site names (kaggle-wmh layout only)",
    )
    p.add_argument(
        "--test-sites",
        type=str,
        default="Amsterdam,Singapore",
        help="Comma-separated site names (kaggle-wmh layout only)",
    )
    p.add_argument(
        "--scanners",
        type=str,
        default="GE3T",
        help=(
            "Comma-separated scanner names to keep; only applies to sites that "
            "have a scanner subdirectory (e.g. Amsterdam). Use 'all' to keep "
            "every scanner."
        ),
    )
    p.add_argument("--max-train", type=int, default=5)
    p.add_argument("--max-test", type=int, default=5)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help=(
            "HD-BET device. 'auto' uses CUDA/ROCm if PyTorch sees it, then MPS, "
            "then CPU."
        ),
    )
    p.add_argument(
        "--enable-tta",
        action="store_true",
        help="Enable HD-BET test-time augmentation. Slower; usually only worth it on GPU.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-trees", type=int, default=20)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--min-cluster-size", type=int, default=10)
    p.add_argument("--training-partitions", type=int, default=4)
    p.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.25,
        help="Class-1 RF probability threshold for raw WMH mask generation.",
    )
    p.add_argument(
        "--spark-master",
        type=str,
        default=None,
        help=(
            "Spark master URL. Default uses local[N] where N is "
            "SLURM_CPUS_PER_TASK when available, otherwise local[4]."
        ),
    )
    p.add_argument(
        "--spark-driver-memory",
        type=str,
        default="6g",
        help="Spark driver memory (default: 6g).",
    )
    p.add_argument(
        "--spark-shuffle-partitions",
        type=int,
        default=8,
        help="Spark SQL shuffle partitions (default: 8).",
    )
    p.add_argument(
        "--skip-skull-strip",
        action="store_true",
        help="Reuse cached HD-BET outputs under <output-root>/skull_stripped",
    )
    p.add_argument(
        "--skull-strip-root",
        type=Path,
        default=None,
        help="Optional existing skull_stripped directory to reuse with --skip-skull-strip.",
    )
    p.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate dataset layout and selected subjects, then exit before HD-BET.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars for logs/non-interactive runs.",
    )
    p.add_argument(
        "--show-subjects",
        action="store_true",
        help="Print selected train/test subject IDs after subsetting.",
    )
    return p.parse_args()


def _comma_split(s: str) -> List[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def _progress_enabled(args: argparse.Namespace) -> bool:
    return not args.no_progress and sys.stderr.isatty()


def _progress(
    items: Iterable[T],
    args: argparse.Namespace,
    desc: str,
    unit: str = "item",
    total: Optional[int] = None,
) -> Iterator[T]:
    """Wrap an iterable in tqdm when this is an interactive run."""
    yield from _progress_bar(items, args, desc=desc, unit=unit, total=total)


def _progress_bar(
    items: Iterable[T],
    args: argparse.Namespace,
    desc: str,
    unit: str = "item",
    total: Optional[int] = None,
):
    if total is None and hasattr(items, "__len__"):
        total = len(items)  # type: ignore[arg-type]
    return tqdm(
        items,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        disable=not _progress_enabled(args),
    )


def _progress_context(args: argparse.Namespace):
    if _progress_enabled(args):
        return logging_redirect_tqdm()
    return nullcontext()


def _scanner_label(record: SubjectRecord) -> str:
    subject_dir = Path(record.flair_path).parent.parent
    scanner_or_site = subject_dir.parent.name
    if record.site and scanner_or_site == record.site:
        return "none"
    return scanner_or_site


def _torch_backend_summary() -> str:
    try:
        import torch
    except Exception as exc:
        return f"torch unavailable: {exc}"

    parts = [f"torch={torch.__version__}"]
    parts.append(f"cuda_available={torch.cuda.is_available()}")
    parts.append(f"cuda_version={torch.version.cuda}")
    parts.append(f"hip_version={getattr(torch.version, 'hip', None)}")
    parts.append(f"cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        parts.append(
            "cuda_devices="
            + ",".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count()))
        )
    if hasattr(torch.backends, "mps"):
        parts.append(f"mps_available={torch.backends.mps.is_available()}")
    return " ".join(parts)


def _resolve_spark_master(args: argparse.Namespace) -> str:
    """Resolve Spark master for local development / single-node Slurm jobs."""
    if args.spark_master:
        return args.spark_master
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit() and int(slurm_cpus) > 0:
        return f"local[{slurm_cpus}]"
    return "local[4]"


def resolve_hdbet_device(requested: str) -> str:
    """Resolve/validate the HD-BET device before launching slow work.

    PyTorch ROCm builds also expose AMD GPUs through the ``cuda`` device API, so
    ``cuda`` means NVIDIA CUDA or AMD ROCm, depending on the installed torch.
    """
    try:
        import torch
    except Exception as exc:
        if requested == "cpu":
            logger.warning("could not inspect torch backend, continuing on CPU: %s", exc)
            return "cpu"
        raise RuntimeError(f"cannot use HD-BET device={requested!r}: torch import failed: {exc}")

    cuda_ok = bool(torch.cuda.is_available())
    mps_ok = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    if requested == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"

    if requested == "cuda" and not cuda_ok:
        raise RuntimeError(
            "HD-BET device=cuda requested, but this venv does not see a CUDA/ROCm "
            f"GPU. Backend: {_torch_backend_summary()}"
        )
    if requested == "mps" and not mps_ok:
        raise RuntimeError(
            "HD-BET device=mps requested, but PyTorch MPS is unavailable. "
            f"Backend: {_torch_backend_summary()}"
        )
    return requested


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_arrange(args: argparse.Namespace, write_manifests: bool = True) -> ArrangedDataset:
    manifests_dir = args.output_root / "manifests"

    if args.layout == "kaggle-wmh":
        scanners = None if args.scanners.lower() == "all" else _comma_split(args.scanners)
        train_discovered = arrange_kaggle_wmh(
            args.data_root, "training", _comma_split(args.train_sites), scanners=scanners
        )
        test_discovered = arrange_kaggle_wmh(
            args.data_root, "test", _comma_split(args.test_sites), scanners=scanners
        )
    else:
        train_discovered = arrange_generic(args.data_root, split="training")
        test_discovered = arrange_generic(args.data_root, split="test")

    train_selected = subset(train_discovered, args.max_train)
    test_selected = subset(test_discovered, args.max_test)

    if write_manifests:
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_manifest(train_selected, manifests_dir / "train_manifest.parquet")
        write_manifest(test_selected, manifests_dir / "test_manifest.parquet")

    logger.info(
        "arranged dataset: train discovered=%d selected=%d; test discovered=%d selected=%d",
        len(train_discovered),
        len(train_selected),
        len(test_discovered),
        len(test_selected),
    )
    return ArrangedDataset(
        train_discovered=train_discovered,
        test_discovered=test_discovered,
        train_selected=train_selected,
        test_selected=test_selected,
        manifests_dir=manifests_dir,
    )


def _format_counter(counter: Counter[str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counter.items())) or "none"


def _log_selected_subjects(split: str, records: List[SubjectRecord]) -> None:
    ids = ", ".join(r.subject_id for r in records)
    logger.info("%s selected subjects (%d): %s", split, len(records), ids)


def validate_dataset_arrangement(
    args: argparse.Namespace,
    arranged: ArrangedDataset,
) -> None:
    """Validate records before launching expensive HD-BET work.

    The check intentionally validates all discovered records after the selected
    site/scanner filters, not just the pilot subset, so a run can fail early
    if the dataset tree is only partially arranged.
    """
    split_data = [
        ("training", arranged.train_discovered, arranged.train_selected),
        ("test", arranged.test_discovered, arranged.test_selected),
    ]
    errors: List[str] = []
    non_binary: List[str] = []

    for split, discovered, selected in split_data:
        site_counts = Counter(r.site or "unknown" for r in discovered)
        scanner_counts = Counter(_scanner_label(r) for r in discovered)
        logger.info(
            "preflight %s: discovered=%d selected=%d sites=[%s] scanners=[%s]",
            split,
            len(discovered),
            len(selected),
            _format_counter(site_counts),
            _format_counter(scanner_counts),
        )
        if args.show_subjects:
            _log_selected_subjects(split, selected)

        ids = [r.subject_id for r in discovered]
        duplicate_ids = sorted({sid for sid in ids if ids.count(sid) > 1})
        if duplicate_ids:
            errors.append(f"{split}: duplicate subject_id values: {duplicate_ids}")

        for record in _progress(discovered, args, desc=f"preflight {split}", unit="subject"):
            required = [
                ("T1", record.t1_path),
                ("FLAIR", record.flair_path),
                ("WMH mask", record.gt_mask_path),
            ]
            missing = [name for name, path in required if path is None or not Path(path).exists()]
            if missing:
                errors.append(
                    f"{split}/{record.subject_id}: missing required files: {', '.join(missing)}"
                )
                continue

            try:
                t1_img = nib.load(record.t1_path)
                flair_img = nib.load(record.flair_path)
                gt_img = nib.load(record.gt_mask_path)  # type: ignore[arg-type]
            except Exception as exc:
                errors.append(f"{split}/{record.subject_id}: failed to load NIfTI: {exc}")
                continue

            if t1_img.shape != flair_img.shape:
                errors.append(
                    f"{split}/{record.subject_id}: T1 shape {t1_img.shape} != "
                    f"FLAIR shape {flair_img.shape}"
                )
            if not np.allclose(t1_img.affine, flair_img.affine, atol=PAIR_AFFINE_ATOL):
                delta = float(np.abs(t1_img.affine - flair_img.affine).max())
                errors.append(
                    f"{split}/{record.subject_id}: T1/FLAIR affine mismatch "
                    f"(max delta {delta:.6f}, tolerance {PAIR_AFFINE_ATOL:g})"
                )
            if flair_img.shape != gt_img.shape:
                errors.append(
                    f"{split}/{record.subject_id}: FLAIR shape {flair_img.shape} != "
                    f"WMH mask shape {gt_img.shape}"
                )
            if not np.allclose(flair_img.affine, gt_img.affine, atol=GT_AFFINE_ATOL):
                delta = float(np.abs(flair_img.affine - gt_img.affine).max())
                errors.append(
                    f"{split}/{record.subject_id}: FLAIR/WMH affine mismatch "
                    f"(max delta {delta:.6f}, tolerance {GT_AFFINE_ATOL:g})"
                )

            labels = sorted(float(v) for v in np.unique(np.asarray(gt_img.dataobj)))
            unexpected = [v for v in labels if v not in (0.0, 1.0)]
            if unexpected:
                non_binary.append(
                    f"{split}/{record.subject_id}: labels={labels}; non-WMH labels "
                    "(including label 2 other pathology) are binarized away"
                )

    if non_binary:
        logger.warning(
            "preflight found %d non-binary GT masks; first examples: %s",
            len(non_binary),
            " | ".join(non_binary[:5]),
        )

    if errors:
        preview = "\n- ".join(errors[:20])
        suffix = f"\n... {len(errors) - 20} more errors" if len(errors) > 20 else ""
        raise RuntimeError(f"dataset preflight failed with {len(errors)} error(s):\n- {preview}{suffix}")

    logger.info("dataset preflight OK: required files, shapes, affines, and IDs are valid")


def stage_skull_strip(
    args: argparse.Namespace,
    train_records: List[SubjectRecord],
    test_records: List[SubjectRecord],
) -> Path:
    skull_root = args.output_root / "skull_stripped"
    if args.skip_skull_strip:
        cached_root = args.skull_strip_root or skull_root
        if cached_root.exists():
            logger.info("--skip-skull-strip: reusing %s", cached_root)
            return cached_root
        raise FileNotFoundError(
            "--skip-skull-strip requested but cached HD-BET outputs were not found: "
            f"{cached_root}"
        )

    skull_root.mkdir(parents=True, exist_ok=True)
    stripper = SkullStripper(
        output_dir=skull_root,
        smoke_test=True,  # structural assertions only — no DSC reference masks
        device=args.hdbet_device,
        disable_tta=not args.enable_tta,
    )
    from wmh_spark.preprocessing.skull_strip import ProcessingLogEntry, RawScan, ScanPair

    pairs = []
    for r in _progress(train_records + test_records, args, desc="build skull-strip queue", unit="subject"):
        pairs.append(
            ScanPair(
                subject_id=r.subject_id,
                t1_scan=RawScan(path=Path(r.t1_path), modality="T1", subject_id=r.subject_id),
                flair_scan=RawScan(
                    path=Path(r.flair_path), modality="FLAIR", subject_id=r.subject_id
                ),
            )
        )

    logger.info(
        "starting skull-strip: subjects=%d device=%s enable_tta=%s",
        len(pairs),
        args.hdbet_device,
        args.enable_tta,
    )
    results = []
    log_entries = []
    accepted = rejected = error_count = 0
    batch_t0 = time.time()
    bar = _progress_bar(pairs, args, desc="skull-strip", unit="subject")
    for pair in bar:
        bar.set_postfix_str(f"{pair.subject_id} ok={accepted} err={error_count}")
        result = stripper.process_pair(pair)
        results.append(result)
        if result.status == "accepted":
            accepted += 1
        elif result.status == "rejected":
            rejected += 1
        else:
            error_count += 1
        bar.set_postfix_str(
            f"{pair.subject_id} ok={accepted} rejected={rejected} err={error_count}"
        )
        log_entries.append(
            ProcessingLogEntry(
                run_id=stripper._run_id,  # same run id used by process_batch logs
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                subject_id=result.subject_id,
                t1_dsc=result.t1_dsc,
                flair_dsc=result.flair_dsc,
                status=result.status,
                elapsed_seconds=result.elapsed_seconds,
                error_message=result.error_message,
            )
        )
    stripper._write_log(log_entries, batch_elapsed=time.time() - batch_t0)

    errors = [r for r in results if r.status != "accepted"]
    if errors:
        msg = "; ".join(f"{r.subject_id}={r.status}({r.error_message})" for r in errors)
        raise RuntimeError(f"skull-strip failed for {len(errors)} subjects: {msg}")
    return skull_root


def _stripped_paths(skull_root: Path, record: SubjectRecord) -> tuple[Path, Path, Path, Path]:
    """Return (t1_bet, flair_bet, flair_mask, spatial_prior) for a subject."""
    subj = skull_root / record.subject_id
    t1_stem = Path(record.t1_path).name.split(".nii")[0]
    flair_stem = Path(record.flair_path).name.split(".nii")[0]
    return (
        subj / f"{t1_stem}_bet.nii.gz",
        subj / f"{flair_stem}_bet.nii.gz",
        subj / f"{flair_stem}_bet_mask.nii.gz",
        subj / "spatial_prior.nii.gz",
    )


def stage_spatial_prior(
    args: argparse.Namespace,
    skull_root: Path,
    records: List[SubjectRecord],
) -> None:
    """Generate per-subject spatial prior + binarised GT mask.

    - spatial_prior.nii.gz = copy of FLAIR brain mask (HD-BET output)
    - wmh_binary.nii.gz = GT mask with non-{0,1} labels coerced to 0
      (WMH-2017 uses label 2 for "other pathology" which is excluded by
       challenge convention). Records without GT keep gt_mask_path=None.
    """
    logger.info("preparing spatial priors and binary GT masks: subjects=%d", len(records))
    for r in _progress(records, args, desc="prepare masks", unit="subject"):
        _, _, flair_mask, prior = _stripped_paths(skull_root, r)
        if not flair_mask.exists():
            raise FileNotFoundError(f"FLAIR brain mask missing: {flair_mask}")
        if not prior.exists():
            shutil.copyfile(flair_mask, prior)

        if r.gt_mask_path is None:
            continue
        binary_path = skull_root / r.subject_id / "wmh_binary.nii.gz"
        if not binary_path.exists():
            img = nib.load(r.gt_mask_path)
            arr = np.asarray(img.dataobj)
            binary = (arr == 1).astype(np.uint8)
            save_volume(binary, img.affine, binary_path, dtype=np.uint8)
        r.gt_mask_path = str(binary_path)


def stage_train(
    spark: SparkSession,
    args: argparse.Namespace,
    skull_root: Path,
    train_records: List[SubjectRecord],
) -> tuple[object, ClassificationConfig, ClassBalanceStats]:
    logger.info("building training feature DataFrames: subjects=%d", len(train_records))
    feature_dfs: List[DataFrame] = []
    for r in _progress(train_records, args, desc="train features", unit="subject"):
        if r.gt_mask_path is None:
            logger.warning("training subject without GT, skipping: %s", r.subject_id)
            continue
        t1_bet, flair_bet, _, prior = _stripped_paths(skull_root, r)
        df = build_feature_dataframe(
            spark=spark,
            subject_id=r.subject_id,
            t1_path=t1_bet,
            flair_path=flair_bet,
            spatial_prior_path=prior,
            mask_path=r.gt_mask_path,
        )
        feature_dfs.append(df)

    if not feature_dfs:
        raise RuntimeError("no training subjects with ground truth — cannot train")

    training = feature_dfs[0]
    for extra in feature_dfs[1:]:
        training = training.unionByName(extra)
    training = training.cache()

    rf_config = ClassificationConfig(
        num_trees=args.num_trees,
        max_depth=args.max_depth,
        training_partitions=args.training_partitions,
        seed=args.seed,
        prediction_threshold=args.prediction_threshold,
    )
    balance_stats = calculate_class_balance_stats(training, rf_config)
    logger.info(
        (
            "training RF: subjects=%d trees=%d max_depth=%d "
            "negative_count=%d positive_count=%d positive_weight=%.4f"
        ),
        len(feature_dfs),
        rf_config.num_trees,
        rf_config.max_depth,
        balance_stats.negative_count,
        balance_stats.positive_count,
        balance_stats.positive_class_weight,
    )
    model = train_random_forest_model(training, rf_config, balance_stats=balance_stats)
    return model, rf_config, balance_stats


def stage_predict_one(
    spark: SparkSession,
    args: argparse.Namespace,
    skull_root: Path,
    record: SubjectRecord,
    model,
    rf_config: ClassificationConfig,
    pred_dir: Path,
):
    """Run inference + postprocess for one subject.

    Returns ``(flair, pred_mask, affine, gt_mask, dice_result_or_None,
    raw_predicted_voxels, max_wmh_probability)``.

    Axis convention (matches build_voxel_dataframe): the DataFrame columns
    'z'/'y'/'x' are the first/second/third axis indices into the NIfTI array,
    so ``flair.shape`` is also ``(z_size, y_size, x_size)`` for the purposes
    of indexing here.
    """
    t1_bet, flair_bet, _, prior = _stripped_paths(skull_root, record)
    feat = build_feature_dataframe(
        spark=spark,
        subject_id=record.subject_id,
        t1_path=t1_bet,
        flair_path=flair_bet,
        spatial_prior_path=prior,
        mask_path=record.gt_mask_path,
    )
    raw = predict_voxel_mask(model, feat, rf_config).cache()
    raw_diag = raw.agg(
        F.sum(F.col(rf_config.mask_column)).alias("raw_predicted_voxels"),
        F.max(F.col(rf_config.positive_probability_column)).alias("max_wmh_probability"),
    ).first()
    raw_predicted_voxels = int(raw_diag["raw_predicted_voxels"] or 0)
    max_wmh_probability = raw_diag["max_wmh_probability"]
    max_wmh_probability = float(max_wmh_probability) if max_wmh_probability is not None else None

    flair_img = nib.load(str(flair_bet))
    flair = np.asarray(flair_img.dataobj, dtype=np.float32)
    affine = flair_img.affine

    post = postprocess_predictions(
        raw,
        PostProcessingConfig(
            volume_shape=flair.shape,
            min_cluster_size=args.min_cluster_size,
        ),
    )

    rows = (
        post.select("z", "y", "x", "postprocessed_mask").where("postprocessed_mask = 1").collect()
    )
    pred = np.zeros(flair.shape, dtype=np.uint8)
    for row in rows:
        pred[int(row["z"]), int(row["y"]), int(row["x"])] = 1

    save_volume(pred, affine, pred_dir / record.subject_id / "wmh_pred.nii.gz", dtype=np.uint8)

    gt = None
    if record.gt_mask_path:
        gt_img = nib.load(record.gt_mask_path)
        gt = (np.asarray(gt_img.dataobj) > 0).astype(np.uint8)

    # Compute Dice via Spark on the postprocessed DataFrame for consistency.
    dice = None
    if "label" in post.columns:
        dice_cfg = EvaluationConfig(
            prediction_column="postprocessed_mask", label_column="label", dsc_threshold=0.0
        )
        try:
            dice = calculate_dice_result(post, dice_cfg)
        except ValueError:
            dice = None

    raw.unpersist()

    return flair, pred, affine, gt, dice, raw_predicted_voxels, max_wmh_probability


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if not 0.0 <= args.prediction_threshold <= 1.0:
        raise ValueError("--prediction-threshold must be between 0 and 1")

    t_start = time.time()

    with _progress_context(args):
        arranged = stage_arrange(args, write_manifests=False)
        validate_dataset_arrangement(args, arranged)

        if args.preflight_only:
            logger.info("preflight-only complete; exiting before skull stripping")
            return 0

        args.hdbet_device = resolve_hdbet_device(args.device)
        logger.info(
            "HD-BET backend: requested=%s resolved=%s %s",
            args.device,
            args.hdbet_device,
            _torch_backend_summary(),
        )

        train_records = arranged.train_selected
        test_records = arranged.test_selected

        args.output_root.mkdir(parents=True, exist_ok=True)
        write_manifest(train_records, arranged.manifests_dir / "train_manifest.parquet")
        write_manifest(test_records, arranged.manifests_dir / "test_manifest.parquet")

        skull_root = stage_skull_strip(args, train_records, test_records)
        stage_spatial_prior(args, skull_root, train_records + test_records)

        spark_master = _resolve_spark_master(args)
        logger.info(
            "starting Spark session: master=%s driver_memory=%s shuffle_partitions=%d",
            spark_master,
            args.spark_driver_memory,
            args.spark_shuffle_partitions,
        )
        spark = (
            SparkSession.builder.appName("wmh-spark-pipeline")
            .master(spark_master)
            .config("spark.driver.memory", args.spark_driver_memory)
            .config("spark.sql.shuffle.partitions", str(args.spark_shuffle_partitions))
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        try:
            model, rf_config, balance_stats = stage_train(spark, args, skull_root, train_records)

            model_dir = args.output_root / "model"
            if model_dir.exists():
                shutil.rmtree(model_dir)
            logger.info("saving RF model to %s", model_dir)
            model.write().overwrite().save(str(model_dir))

            per_subject_dir = args.output_root / "per_subject"
            pred_dir = args.output_root / "predictions"
            bench_dir = args.output_root / "benchmarks"
            bench_dir.mkdir(parents=True, exist_ok=True)

            all_metrics: List[SubjectMetrics] = []
            logger.info("running prediction, postprocessing, and reports: subjects=%d", len(test_records))

            for r in _progress(test_records, args, desc="predict/report", unit="subject"):
                t0 = time.time()
                (
                    flair,
                    pred,
                    affine,
                    gt,
                    dice,
                    raw_predicted_voxels,
                    max_wmh_probability,
                ) = stage_predict_one(
                    spark, args, skull_root, r, model, rf_config, pred_dir
                )
                vox_mm3 = voxel_volume_mm3(affine)

                lesion_count = count_lesions(pred)
                predicted_voxels = int(pred.sum())
                predicted_volume = predicted_voxels * vox_mm3
                reference_voxels = int(gt.sum()) if gt is not None else None
                reference_volume = reference_voxels * vox_mm3 if reference_voxels is not None else None

                metrics = SubjectMetrics(
                    subject_id=r.subject_id,
                    site=r.site,
                    dsc=dice.dsc if dice else None,
                    predicted_voxels=predicted_voxels,
                    reference_voxels=reference_voxels,
                    intersection=dice.intersection if dice else None,
                    lesion_count=lesion_count,
                    predicted_volume_mm3=predicted_volume,
                    reference_volume_mm3=reference_volume,
                    voxel_volume_mm3=vox_mm3,
                    raw_predicted_voxels=raw_predicted_voxels,
                    max_wmh_probability=max_wmh_probability,
                    prediction_threshold=rf_config.prediction_threshold,
                )
                all_metrics.append(metrics)

                subj_out = per_subject_dir / r.subject_id
                render_subject_overlay(
                    flair_path=Path(_stripped_paths(skull_root, r)[1]),
                    pred_mask=pred,
                    gt_mask=gt,
                    out_png=subj_out / "overlay.png",
                )
                write_subject_report(metrics, subj_out)

                elapsed = time.time() - t0
                log_subject_benchmark(
                    subject_id=r.subject_id,
                    elapsed_seconds=elapsed,
                    matlab_baseline_seconds=120.0,
                    worker_nodes=4,
                    voxel_count=int(np.prod(flair.shape)),
                    output_path=bench_dir / "subject_bench.jsonl",
                )
                logger.info(
                    (
                        "[%s] dsc=%s lesions=%d raw_pred_vox=%d pred_vox=%d "
                        "max_prob=%s elapsed=%.1fs"
                    ),
                    r.subject_id,
                    f"{dice.dsc:.4f}" if dice else "n/a",
                    lesion_count,
                    raw_predicted_voxels,
                    predicted_voxels,
                    f"{max_wmh_probability:.4f}" if max_wmh_probability is not None else "n/a",
                    elapsed,
                )

            for _ in _progress([None], args, desc="aggregate report", unit="stage"):
                write_aggregate_report(all_metrics, args.output_root / "aggregate")

            summary = {
                "subjects_train": len(train_records),
                "subjects_test": len(test_records),
                "mean_dsc": float(np.mean([m.dsc for m in all_metrics if m.dsc is not None]))
                if any(m.dsc is not None for m in all_metrics)
                else None,
                "total_elapsed_seconds": round(time.time() - t_start, 2),
                "rf_num_trees": rf_config.num_trees,
                "rf_max_depth": rf_config.max_depth,
                "min_cluster_size": args.min_cluster_size,
                "prediction_threshold": rf_config.prediction_threshold,
                "class_weighting": True,
                "positive_class_weight": balance_stats.positive_class_weight,
                "positive_class_weight_cap": rf_config.positive_class_weight_cap,
                "training_positive_count": balance_stats.positive_count,
                "training_negative_count": balance_stats.negative_count,
            }
            (bench_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            logger.info("DONE: %s", json.dumps(summary))

        finally:
            spark.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
