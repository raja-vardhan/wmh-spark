"""Scan a data directory and emit a Parquet manifest of subjects.

Expected layout (Kaggle WMH Segmentation Dataset and WMH Challenge both fit):

    data_dir/
      subject_001/
        flair.nii.gz
        t1.nii.gz
        wmh_mask.nii.gz       # ground truth (optional)
        ubo_mask.nii.gz       # secondary reference (optional)
      subject_002/
        ...

Run:
    python scripts/make_manifest.py \\
        --data-dir data/raw \\
        --out      data/manifest.parquet \\
        --split-seed 42 \\
        --train 0.7 --val 0.15 --test 0.15

The manifest is the single source of truth for which subjects the pipeline
processes. Train/val/test split is deterministic given the seed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _find(subj_dir: Path, *candidates: str) -> str | None:
    for name in candidates:
        p = subj_dir / name
        if p.exists():
            return str(p)
    return None


def build_manifest(data_dir: Path) -> pd.DataFrame:
    rows = []
    for subj_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        flair = _find(subj_dir, "flair.nii.gz", "FLAIR.nii.gz", "flair.nii")
        t1 = _find(subj_dir, "t1.nii.gz", "T1.nii.gz", "t1.nii")
        if not flair or not t1:
            continue
        gt = _find(subj_dir, "wmh_mask.nii.gz", "wmh.nii.gz", "ground_truth.nii.gz")
        ubo = _find(subj_dir, "ubo_mask.nii.gz", "ubo.nii.gz")
        rows.append({
            "subject_id": subj_dir.name,
            "flair_path": flair,
            "t1_path": t1,
            "gt_mask_path": gt,
            "ubo_mask_path": ubo,
            "site": None,
            "age": None,
            "split": None,
        })
    if not rows:
        raise RuntimeError(f"No subjects found under {data_dir}")
    return pd.DataFrame(rows)


def assign_splits(df: pd.DataFrame, seed: int, train: float, val: float, test: float):
    """Reproducible train/val/test split. Persist to manifest."""
    assert abs(train + val + test - 1.0) < 1e-6, "splits must sum to 1"
    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).index.tolist()
    n = len(df)
    n_train = int(n * train)
    n_val = int(n * val)
    splits = ["test"] * n
    for i in rng[:n_train]:
        splits[i] = "train"
    for i in rng[n_train:n_train + n_val]:
        splits[i] = "val"
    df = df.copy()
    df["split"] = splits
    return df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    args = parser.parse_args()

    df = build_manifest(Path(args.data_dir))
    df = assign_splits(df, args.split_seed, args.train, args.val, args.test)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} subjects to {out_path}")
    print(df["split"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
