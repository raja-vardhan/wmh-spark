"""IO helpers for NIfTI volumes and the subject manifest.

The manifest is a Parquet table with one row per subject. Spark reads it as
a DataFrame, then we ``rdd.map`` over the rows so each worker pulls only the
volumes assigned to it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


@dataclass
class SubjectRecord:
    """One row of the subject manifest.

    Stored as Parquet so Spark can read partitions in parallel. Paths are kept
    as strings (Parquet doesn't have a native Path type) and resolved on each
    worker. We deliberately don't ship volumes through the manifest -- only
    pointers.
    """

    subject_id: str
    flair_path: str
    t1_path: str
    gt_mask_path: Optional[str] = None
    ubo_mask_path: Optional[str] = None
    site: Optional[str] = None
    age: Optional[float] = None
    split: Optional[str] = None  # "train" | "val" | "test"

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "flair_path": self.flair_path,
            "t1_path": self.t1_path,
            "gt_mask_path": self.gt_mask_path,
            "ubo_mask_path": self.ubo_mask_path,
            "site": self.site,
            "age": self.age,
            "split": self.split,
        }


# ---------------------------------------------------------------------------
# Volume IO
# ---------------------------------------------------------------------------


def load_volume(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return ``(data, affine)``.

    We always cast to float32. The ``affine`` is needed to write outputs back
    in the same coordinate system, which is essential for downstream registration.
    """
    img = nib.load(str(path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.affine


def save_volume(
    data: np.ndarray,
    affine: np.ndarray,
    path: str | Path,
    dtype: type = np.float32,
) -> None:
    """Write a volume to disk, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(dtype), affine)
    nib.save(img, str(path))


def load_mask(path: str | Path) -> np.ndarray:
    """Load a binary mask (any nonzero voxel is foreground)."""
    data, _ = load_volume(path)
    return (data > 0).astype(np.uint8)


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


def write_checksum_manifest(records: Iterable[SubjectRecord], out_path: str | Path) -> None:
    """Emit a tab-separated checksum file: one line per (subject, file)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in records:
            for kind in ("flair_path", "t1_path", "gt_mask_path"):
                p = getattr(rec, kind)
                if p and Path(p).exists():
                    f.write(f"{rec.subject_id}\t{kind}\t{sha256_of_file(p)}\t{p}\n")
