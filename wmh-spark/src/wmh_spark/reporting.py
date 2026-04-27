"""Per-subject clinical overlays and aggregate report generation.

The per-subject report is a single HTML page containing a 3x3 axial mosaic
of the FLAIR volume with the predicted WMH mask painted in red and the
ground-truth mask outlined in green, plus a stats table (Dice, lesion count,
predicted/reference volume in mm^3). The aggregate report links to each
subject report and shows per-site DSC and predicted-vs-true volume scatter.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SubjectMetrics:
    subject_id: str
    site: Optional[str]
    dsc: Optional[float]
    predicted_voxels: int
    reference_voxels: Optional[int]
    intersection: Optional[int]
    lesion_count: int
    predicted_volume_mm3: float
    reference_volume_mm3: Optional[float]
    voxel_volume_mm3: float
    raw_predicted_voxels: Optional[int] = None
    max_wmh_probability: Optional[float] = None
    prediction_threshold: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "subject_id": self.subject_id,
            "site": self.site,
            "dsc": self.dsc,
            "raw_predicted_voxels": self.raw_predicted_voxels,
            "predicted_voxels": self.predicted_voxels,
            "reference_voxels": self.reference_voxels,
            "intersection": self.intersection,
            "lesion_count": self.lesion_count,
            "predicted_volume_mm3": self.predicted_volume_mm3,
            "reference_volume_mm3": self.reference_volume_mm3,
            "voxel_volume_mm3": self.voxel_volume_mm3,
            "max_wmh_probability": self.max_wmh_probability,
            "prediction_threshold": self.prediction_threshold,
        }


def voxel_volume_mm3(affine: np.ndarray) -> float:
    """Return the per-voxel volume in mm^3 from a NIfTI affine."""
    return float(abs(np.linalg.det(affine[:3, :3])))


def _select_axial_slices(mask_or_flair: np.ndarray, count: int = 9) -> List[int]:
    """Pick `count` axial slice indices, weighted toward slices with most lesion."""
    nz = mask_or_flair.shape[2]
    if mask_or_flair.dtype == bool or set(np.unique(mask_or_flair)).issubset({0, 1}):
        per_slice = mask_or_flair.sum(axis=(0, 1))
        if per_slice.sum() > 0:
            top = np.argsort(per_slice)[::-1][: max(count, 1)]
            return sorted(int(i) for i in top)[:count] or [nz // 2]
    # fall back to evenly spaced slices through the central 60% of the volume
    lo, hi = int(nz * 0.2), int(nz * 0.8)
    return [int(round(v)) for v in np.linspace(lo, hi - 1, count)]


def render_subject_overlay(
    flair_path: Path,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    out_png: Path,
    n_slices: int = 9,
) -> None:
    """Render a 3x3 axial mosaic of FLAIR with prediction and GT overlays.

    Predictions are filled in red at alpha=0.5; ground truth is drawn as a
    green contour line so both layers are visible on overlap.
    """
    img = nib.load(str(flair_path))
    flair = np.asarray(img.dataobj, dtype=np.float32)
    if pred_mask.shape != flair.shape:
        raise ValueError(
            f"pred_mask shape {pred_mask.shape} != flair shape {flair.shape}"
        )

    target = gt_mask if gt_mask is not None and gt_mask.any() else pred_mask
    slice_indices = _select_axial_slices(
        target if target.any() else flair, count=n_slices
    )

    cols = 3
    rows = (n_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)

    vmax = float(np.percentile(flair[flair > 0], 99)) if (flair > 0).any() else 1.0

    for ax_idx in range(rows * cols):
        ax = axes[ax_idx // cols, ax_idx % cols]
        ax.set_axis_off()
        if ax_idx >= len(slice_indices):
            continue
        z = slice_indices[ax_idx]
        ax.imshow(np.rot90(flair[:, :, z]), cmap="gray", vmin=0, vmax=vmax)

        pred_slice = np.rot90(pred_mask[:, :, z])
        if pred_slice.any():
            overlay = np.zeros((*pred_slice.shape, 4), dtype=np.float32)
            overlay[pred_slice > 0] = (1.0, 0.0, 0.0, 0.5)  # red, alpha 0.5
            ax.imshow(overlay)

        if gt_mask is not None:
            gt_slice = np.rot90(gt_mask[:, :, z])
            if gt_slice.any():
                ax.contour(gt_slice, levels=[0.5], colors="lime", linewidths=0.6)

        ax.set_title(f"z={z}", fontsize=8, color="white", backgroundcolor="black")

    fig.suptitle(
        f"FLAIR + Predicted WMH (red)"
        + (" + GT outline (green)" if gt_mask is not None else ""),
        fontsize=10,
    )
    fig.subplots_adjust(top=0.95, wspace=0.02, hspace=0.05)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight", facecolor="black")
    plt.close(fig)


_SUBJECT_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>{subject_id} — WMH report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 24px; max-width: 900px; }}
  h1 {{ font-size: 1.4em; }}
  table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
  th {{ background: #f4f4f4; }}
  img {{ max-width: 100%; border: 1px solid #ddd; }}
  .muted {{ color: #888; }}
</style></head><body>
<h1>{subject_id}</h1>
<p class="muted">Site: {site} &nbsp;·&nbsp; Voxel volume: {vox:.3f} mm³</p>
<img src="overlay.png" alt="FLAIR overlay">
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Dice (DSC)</td><td>{dsc}</td></tr>
  <tr><td>Raw predicted lesion voxels</td><td>{raw_pv}</td></tr>
  <tr><td>Predicted lesion voxels</td><td>{pv:,}</td></tr>
  <tr><td>Reference lesion voxels</td><td>{rv}</td></tr>
  <tr><td>Predicted lesion clusters</td><td>{lc}</td></tr>
  <tr><td>Predicted volume (mm³)</td><td>{pvol:,.1f}</td></tr>
  <tr><td>Reference volume (mm³)</td><td>{rvol}</td></tr>
  <tr><td>Max WMH probability</td><td>{max_prob}</td></tr>
  <tr><td>Prediction threshold</td><td>{pred_threshold}</td></tr>
</table>
<p class="muted">Red = predicted WMH (α=0.5). Green outline = ground truth.</p>
</body></html>
"""


def _fmt_optional(value, fmt: str) -> str:
    if value is None:
        return "—"
    return format(value, fmt)


def write_subject_report(metrics: SubjectMetrics, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(metrics.to_dict(), indent=2))
    html_path = out_dir / "report.html"
    html_path.write_text(
        _SUBJECT_HTML.format(
            subject_id=metrics.subject_id,
            site=metrics.site or "—",
            vox=metrics.voxel_volume_mm3,
            dsc=_fmt_optional(metrics.dsc, ".4f"),
            raw_pv=_fmt_optional(metrics.raw_predicted_voxels, ",d"),
            pv=metrics.predicted_voxels,
            rv=_fmt_optional(metrics.reference_voxels, ",d"),
            lc=metrics.lesion_count,
            pvol=metrics.predicted_volume_mm3,
            rvol=_fmt_optional(metrics.reference_volume_mm3, ",.1f"),
            max_prob=_fmt_optional(metrics.max_wmh_probability, ".4f"),
            pred_threshold=_fmt_optional(metrics.prediction_threshold, ".4f"),
        )
    )
    return html_path


def write_aggregate_report(
    metrics: List[SubjectMetrics],
    out_dir: Path,
    per_subject_subdir: str = "../per_subject",
) -> Path:
    """Write CSV + per-site DSC chart + volume scatter + index HTML."""
    if not metrics:
        raise ValueError("write_aggregate_report: no metrics provided")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "metrics.csv"
    columns = list(metrics[0].to_dict().keys())
    with open(csv_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for m in metrics:
            row = m.to_dict()
            f.write(",".join(_csv_cell(row[c]) for c in columns) + "\n")

    # Per-site DSC bar chart
    by_site: dict[str, list[float]] = {}
    for m in metrics:
        if m.dsc is None:
            continue
        by_site.setdefault(m.site or "unknown", []).append(m.dsc)

    if by_site:
        sites = sorted(by_site)
        means = [float(np.mean(by_site[s])) for s in sites]
        stds = [float(np.std(by_site[s])) for s in sites]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(sites, means, yerr=stds, capsize=4, color="#3a76b8")
        ax.set_ylabel("Dice (mean ± std)")
        ax.set_ylim(0, 1)
        ax.set_title("Per-site Dice")
        for i, s in enumerate(sites):
            ax.text(i, means[i] + 0.02, f"n={len(by_site[s])}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "per_site_dsc.png", dpi=110)
        plt.close(fig)

    # Volume scatter
    if any(m.reference_volume_mm3 is not None for m in metrics):
        xs = [m.reference_volume_mm3 for m in metrics if m.reference_volume_mm3 is not None]
        ys = [m.predicted_volume_mm3 for m in metrics if m.reference_volume_mm3 is not None]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xs, ys, color="#cc4422")
        lim = max(max(xs, default=1.0), max(ys, default=1.0)) * 1.05
        ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Reference WMH volume (mm³)")
        ax.set_ylabel("Predicted WMH volume (mm³)")
        ax.set_title("Predicted vs reference lesion volume")
        fig.tight_layout()
        fig.savefig(out_dir / "volume_scatter.png", dpi=110)
        plt.close(fig)

    # Index HTML
    rows = []
    for m in metrics:
        rows.append(
            "<tr>"
            f"<td><a href='{per_subject_subdir}/{m.subject_id}/report.html'>{m.subject_id}</a></td>"
            f"<td>{m.site or '—'}</td>"
            f"<td>{_fmt_optional(m.dsc, '.4f')}</td>"
            f"<td>{_fmt_optional(m.raw_predicted_voxels, ',d')}</td>"
            f"<td>{m.lesion_count}</td>"
            f"<td>{m.predicted_volume_mm3:,.1f}</td>"
            f"<td>{_fmt_optional(m.reference_volume_mm3, ',.1f')}</td>"
            "</tr>"
        )
    summary = _summary_block(metrics)
    images = _images_block(out_dir)
    html_path = out_dir / "report.html"
    html_path.write_text(
        f"""<!doctype html>
<html><head><meta charset="utf-8"><title>WMH run — aggregate report</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 24px; max-width: 1100px; }}
  h1 {{ font-size: 1.5em; }}
  h2 {{ font-size: 1.1em; margin-top: 24px; }}
  table {{ border-collapse: collapse; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; }}
  th {{ background: #f4f4f4; }}
  .images {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .images img {{ max-height: 360px; border: 1px solid #ddd; }}
</style></head><body>
<h1>WMH pipeline run — aggregate report</h1>
{summary}
<h2>Plots</h2>
<div class="images">{images}</div>
<h2>Per-subject results</h2>
<table>
  <tr><th>Subject</th><th>Site</th><th>Dice</th><th>Raw voxels</th><th>Lesions</th>
      <th>Predicted vol (mm³)</th><th>Reference vol (mm³)</th></tr>
  {''.join(rows)}
</table>
</body></html>
"""
    )
    return html_path


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _summary_block(metrics: List[SubjectMetrics]) -> str:
    dsc_values = [m.dsc for m in metrics if m.dsc is not None]
    lines = [f"<p>Subjects: <b>{len(metrics)}</b></p>"]
    if dsc_values:
        lines.append(
            f"<p>Mean Dice: <b>{np.mean(dsc_values):.4f}</b> "
            f"(median {np.median(dsc_values):.4f}, "
            f"min {min(dsc_values):.4f}, max {max(dsc_values):.4f})</p>"
        )
    return "".join(lines)


def _images_block(out_dir: Path) -> str:
    parts: list[str] = []
    for name in ("per_site_dsc.png", "volume_scatter.png"):
        if (out_dir / name).exists():
            parts.append(f"<img src='{name}' alt='{name}'>")
    return "".join(parts)


def count_lesions(mask: np.ndarray) -> int:
    """Count 26-connected lesion clusters in a 3D binary mask."""
    from scipy import ndimage

    if mask.ndim != 3:
        raise ValueError("count_lesions requires a 3D volume")
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    _, n = ndimage.label(mask.astype(bool), structure=structure)
    return int(n)
