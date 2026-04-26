"""I/O primitives for NIfTI volumes and the subject manifest.

Phase 1 implementation — local smoke-test scope.
See build_voxel_dataframe for the known Phase 1 architectural trade-off
and the T032 TODO for the CHPC-scale replacement.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


@dataclass
class SubjectRecord:
    """One row of the subject manifest (stored as Parquet)."""

    subject_id: str
    t1_path: str
    flair_path: str
    gt_mask_path: Optional[str] = None
    site: Optional[str] = None
    age: Optional[float] = None
    split: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "t1_path": self.t1_path,
            "flair_path": self.flair_path,
            "gt_mask_path": self.gt_mask_path,
            "site": self.site,
            "age": self.age,
            "split": self.split,
        }


# ---------------------------------------------------------------------------
# Volume I/O primitives
# ---------------------------------------------------------------------------


def load_volume(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return (data float32, affine).

    Raises FileNotFoundError if the path does not exist.
    Wraps nibabel errors with the file path for easier debugging.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    try:
        img = nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.float32), img.affine
    except Exception as exc:
        raise IOError(f"Failed to load NIfTI file {path}: {exc}") from exc


def save_volume(
    data: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
    dtype: type = np.float32,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(dtype), affine), str(path))


def load_mask(path: str | Path) -> np.ndarray:
    """Load a NIfTI mask; any nonzero voxel is treated as foreground."""
    data, _ = load_volume(path)
    return (data > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_subject_volumes(
    t1_path: str | Path,
    flair_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate paired T1 + FLAIR volumes.

    Returns (t1_data, flair_data, affine).
    Raises ValueError on shape or affine mismatch.
    """
    t1, t1_affine = load_volume(t1_path)
    flair, flair_affine = load_volume(flair_path)

    if t1.shape != flair.shape:
        raise ValueError(
            f"dimension mismatch: T1 shape {t1.shape} != FLAIR shape {flair.shape}"
        )
    if not np.allclose(t1_affine, flair_affine, atol=1e-4):
        raise ValueError(
            f"Affine mismatch between T1 and FLAIR — volumes must be co-registered "
            f"before ingestion. Max delta: {np.abs(t1_affine - flair_affine).max():.6f}"
        )
    return t1, flair, t1_affine


def validate_binary_mask(mask_path: str | Path) -> np.ndarray:
    """Load and validate a binary lesion mask (values must be 0 or 1 only).

    Returns the mask array as uint8.
    Raises ValueError if unexpected values are found.
    """
    data, _ = load_volume(mask_path)
    unique = set(np.unique(data).tolist())
    allowed = {0.0, 1.0, 0, 1}
    unexpected = unique - allowed
    if unexpected:
        raise ValueError(
            f"non-binary mask: expected only {{0, 1}}, found values {sorted(unexpected)} "
            f"in {mask_path}"
        )
    return data.astype(np.uint8)


# ---------------------------------------------------------------------------
# Partition planning (Constitution Principle III)
# ---------------------------------------------------------------------------


def derive_partition_count(
    voxel_count: int,
    bytes_per_row: int = 28,
    max_partition_bytes: int = 16 * 1024**3,
    override: int = 0,
) -> int:
    """Compute a safe partition count respecting the 16 GB/worker ceiling.

    Args:
        voxel_count: Total number of voxels in the DataFrame.
        bytes_per_row: Estimated serialised bytes per voxel row (~28 bytes).
        max_partition_bytes: Per-worker memory ceiling (default 16 GB).
        override: If > 0, return this value directly without computing.
    """
    if override > 0:
        logger.debug("partition_count override=%d", override)
        return override
    count = max(1, math.ceil(voxel_count * bytes_per_row / max_partition_bytes))
    logger.info(
        "auto-derived partition_count=%d (voxels=%d, bytes/row=%d)",
        count, voxel_count, bytes_per_row,
    )
    return count


# ---------------------------------------------------------------------------
# Core DataFrame builder
# ---------------------------------------------------------------------------


def build_voxel_dataframe(
    spark: "SparkSession",
    subject_id: str,
    t1_path: str | Path,
    flair_path: str | Path,
    partition_count: int = 0,
    mask_path: Optional[str | Path] = None,
) -> "DataFrame":
    """Flatten paired NIfTI volumes into a partitioned Spark DataFrame.

    Each output row represents one (x, y, z) voxel with columns:
        subject_id, x, y, z, t1, flair [, label]

    Phase 1 trade-off: numpy arrays are flattened in driver memory before
    spark.createDataFrame(). For a 256³ brain this materialises ~320 MB on
    the driver — acceptable for local smoke tests.
    TODO (T032 / Phase 2): Replace with rdd.mapPartitions + per-worker
    nibabel reads to eliminate driver materialisation and satisfy
    Constitution Principle III at CHPC scale.

    Partition strategy: partition_count=0 auto-derives a safe value from
    voxel count and 16 GB/worker ceiling (Constitution Principle III).
    The resolved count is logged and documented at the .repartition() call.
    """
    import pandas as pd  # imported here to keep module importable without pandas

    t0 = time.time()

    t1, flair, _ = validate_subject_volumes(t1_path, flair_path)
    voxel_count = t1.size

    label_col: Optional[np.ndarray] = None
    if mask_path is not None:
        label_col = validate_binary_mask(mask_path).ravel()

    z_idx, y_idx, x_idx = np.indices(t1.shape, dtype=np.int32)

    pdf = pd.DataFrame(
        {
            "subject_id": subject_id,
            "x": x_idx.ravel(),
            "y": y_idx.ravel(),
            "z": z_idx.ravel(),
            "t1": t1.ravel(),
            "flair": flair.ravel(),
        }
    )
    if label_col is not None:
        pdf["label"] = label_col.astype(np.int32)

    n_parts = derive_partition_count(voxel_count, override=partition_count)

    df = (
        spark.createDataFrame(pdf)
        # Repartition to n_parts — derived from voxel count and 16 GB/worker
        # ceiling (Constitution Principle III). Documented here per convention.
        .repartition(n_parts)
    )

    elapsed = time.time() - t0
    logger.info(
        "build_voxel_dataframe: subject=%s voxels=%d partitions=%d elapsed=%.2fs",
        subject_id, voxel_count, n_parts, elapsed,
    )
    return df


# ---------------------------------------------------------------------------
# Batch ingestion (FR-009)
# ---------------------------------------------------------------------------


def ingest_from_manifest(
    spark: "SparkSession",
    manifest_path: str | Path,
    partition_count: int = 0,
) -> List[Tuple[str, "DataFrame"]]:
    """Read a Parquet manifest and ingest each subject into a voxel DataFrame.

    Driver reads the manifest (coordination only — rows contain paths, not
    volume data). Each volume is loaded and distributed via build_voxel_dataframe.

    Returns a list of (subject_id, DataFrame) tuples.
    """
    import pandas as pd

    manifest = pd.read_parquet(str(manifest_path))
    results = []
    for _, row in manifest.iterrows():
        df = build_voxel_dataframe(
            spark=spark,
            subject_id=row["subject_id"],
            t1_path=row["t1_path"],
            flair_path=row["flair_path"],
            partition_count=partition_count,
            mask_path=row.get("gt_mask_path") or None,
        )
        results.append((row["subject_id"], df))
    return results


# ---------------------------------------------------------------------------
# Checksums (reproducibility)
# ---------------------------------------------------------------------------


def sha256_of_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Return SHA-256 hex digest of a file, streaming in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_checksum_manifest(
    records: Iterable[SubjectRecord], out_path: str | Path
) -> None:
    """Write a TSV checksum file: subject_id, field, sha256, path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            for field in ("t1_path", "flair_path", "gt_mask_path"):
                p = getattr(rec, field)
                if p and Path(p).exists():
                    f.write(f"{rec.subject_id}\t{field}\t{sha256_of_file(p)}\t{p}\n")
