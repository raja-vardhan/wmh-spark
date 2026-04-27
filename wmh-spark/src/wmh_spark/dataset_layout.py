"""Dataset-layout shapers that produce SubjectRecord lists for the manifest.

Two layouts are supported today:

- ``kaggle-wmh``: the WMH 2017 challenge tree
  ``<root>/{training,test}/<site>/[scanner]/<subject_id>/pre/{T1.nii,FLAIR.nii}``
  with ground truth at ``<subject_id>/wmh.nii``. Amsterdam interposes a scanner
  directory (``GE3T``, ``GE1T5``, ``Philips_VU .PETMR_01.``) between site and
  subject; Singapore and Utrecht do not.
- ``generic``: a flat tree of subject directories matching the contract used
  by ``scripts/make_manifest.py`` (``t1*.nii*`` + ``flair*.nii*`` + optional
  ``wmh_mask*.nii*``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from wmh_spark.io_utils import SubjectRecord

logger = logging.getLogger(__name__)


_MANIFEST_SCHEMA = pa.schema(
    [
        ("subject_id", pa.string()),
        ("t1_path", pa.string()),
        ("flair_path", pa.string()),
        ("gt_mask_path", pa.string()),
        ("site", pa.string()),
        ("age", pa.float64()),
        ("split", pa.string()),
    ]
)


def _site_subdirs(site_root: Path) -> List[tuple[Optional[str], Path]]:
    """Return ``(scanner, subject_dir)`` tuples under a site root.

    Some Kaggle sites (Amsterdam) interpose a scanner directory; others do
    not. A directory is treated as a subject iff it contains ``pre/FLAIR.nii``.
    """
    out: List[tuple[Optional[str], Path]] = []
    for entry in sorted(site_root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "pre" / "FLAIR.nii").exists():
            out.append((None, entry))
        else:
            for nested in sorted(entry.iterdir()):
                if nested.is_dir() and (nested / "pre" / "FLAIR.nii").exists():
                    out.append((entry.name, nested))
    return out


def arrange_kaggle_wmh(
    root: Path,
    split: str,
    sites: Iterable[str],
    scanners: Optional[Iterable[str]] = None,
) -> List[SubjectRecord]:
    """Walk a Kaggle WMH-2017 tree and return SubjectRecords for one split.

    Args:
        root: Path containing ``training/`` and ``test/`` subdirectories.
        split: ``"training"`` or ``"test"`` — also propagated to the record.
        sites: Iterable of site names to include (e.g. ``["Amsterdam", "Singapore"]``).

    Subject directories without a ``wmh.nii`` are skipped and logged.
    """
    if split not in {"training", "test"}:
        raise ValueError(f"split must be 'training' or 'test', got {split!r}")

    split_root = root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_root}")

    scanner_filter = {s for s in scanners} if scanners else None

    records: List[SubjectRecord] = []
    seen_ids: set[str] = set()

    for site in sites:
        site_root = split_root / site
        if not site_root.is_dir():
            logger.warning("site directory missing, skipping: %s", site_root)
            continue

        for scanner, subj_dir in _site_subdirs(site_root):
            if scanner_filter and scanner is not None and scanner not in scanner_filter:
                continue

            t1 = subj_dir / "pre" / "T1.nii"
            flair = subj_dir / "pre" / "FLAIR.nii"
            wmh = subj_dir / "wmh.nii"

            if not (t1.exists() and flair.exists()):
                logger.warning("incomplete pair, skipping: %s", subj_dir)
                continue

            parts = [site]
            if scanner:
                parts.append(scanner.replace(" ", "").replace(".", ""))
            parts.append(subj_dir.name)
            subject_id = "_".join(parts)

            if subject_id in seen_ids:
                raise ValueError(f"duplicate subject_id derived: {subject_id}")
            seen_ids.add(subject_id)

            records.append(
                SubjectRecord(
                    subject_id=subject_id,
                    t1_path=str(t1),
                    flair_path=str(flair),
                    gt_mask_path=str(wmh) if wmh.exists() else None,
                    site=site,
                    split=split,
                )
            )

    if not records:
        raise ValueError(
            f"no subjects discovered in {split_root} for sites={list(sites)}"
        )

    logger.info(
        "arrange_kaggle_wmh: split=%s sites=%s subjects=%d",
        split, list(sites), len(records),
    )
    return records


def arrange_generic(root: Path, split: Optional[str] = None) -> List[SubjectRecord]:
    """Discover subjects under a flat layout (one folder per subject)."""
    if not root.is_dir():
        raise FileNotFoundError(f"data-root not found: {root}")

    records: List[SubjectRecord] = []
    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        t1 = sorted(subj_dir.glob("t1*.nii*")) + sorted(subj_dir.glob("T1*.nii*"))
        flair = sorted(subj_dir.glob("flair*.nii*")) + sorted(subj_dir.glob("FLAIR*.nii*"))
        if not t1 or not flair:
            continue
        wmh = sorted(subj_dir.glob("wmh*.nii*"))
        records.append(
            SubjectRecord(
                subject_id=subj_dir.name,
                t1_path=str(t1[0]),
                flair_path=str(flair[0]),
                gt_mask_path=str(wmh[0]) if wmh else None,
                split=split,
            )
        )
    if not records:
        raise ValueError(f"no subjects found under {root}")
    return records


def write_manifest(records: List[SubjectRecord], output: Path) -> None:
    rows = [r.to_dict() for r in records]
    table = pa.Table.from_pylist(rows, schema=_MANIFEST_SCHEMA)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output))
    logger.info("wrote manifest: %s (%d subjects)", output, len(records))


def subset(records: List[SubjectRecord], max_count: Optional[int]) -> List[SubjectRecord]:
    """Stratified subset that preserves site balance when possible."""
    if max_count is None or max_count >= len(records):
        return list(records)
    if max_count <= 0:
        return []

    by_site: dict[str, list[SubjectRecord]] = {}
    for r in records:
        by_site.setdefault(r.site or "_", []).append(r)

    sites = sorted(by_site)
    out: List[SubjectRecord] = []
    i = 0
    while len(out) < max_count:
        progress = False
        for site in sites:
            bucket = by_site[site]
            if i < len(bucket) and len(out) < max_count:
                out.append(bucket[i])
                progress = True
        if not progress:
            break
        i += 1
    return out
