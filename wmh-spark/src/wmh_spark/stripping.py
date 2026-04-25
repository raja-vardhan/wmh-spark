"""Skull-stripping stage.

We support three modes:

1. ``hdbet``       -- HD-BET via Singularity. GPU-preferred. Most accurate.
2. ``synthstrip``  -- FreeSurfer's SynthStrip via Singularity. CPU-fast fallback.
3. ``precomputed`` -- assume ``<input>_brain.nii.gz`` already exists.

Crucial design point: skull-stripping runs as a *separate pre-stage*, not
inside the timed Spark pipeline. This keeps benchmarks honest -- the Spark
vs. MATLAB comparison should not be dominated by HD-BET CPU inference.

Each subject is independent, so this stage maps trivially over an RDD.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import StrippingConfig

logger = logging.getLogger(__name__)


@dataclass
class StripResult:
    """Outcome of skull stripping for one subject.

    We return paths rather than ndarrays because (a) workers shouldn't ship
    full volumes back to the driver, and (b) downstream stages re-load from
    the cached file anyway.
    """

    subject_id: str
    flair_brain_path: str
    t1_brain_path: str
    brain_mask_path: str
    seconds: float
    success: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def strip_subject(
    subject_id: str,
    flair_path: str,
    t1_path: str,
    out_dir: str,
    cfg: StrippingConfig,
) -> StripResult:
    """Skull-strip both modalities for one subject.

    Designed to be called from a worker via ``rdd.map``. All exceptions are
    caught and returned as a failed ``StripResult`` so a bad subject doesn't
    kill the whole job.
    """
    import time

    t0 = time.perf_counter()
    out_dir_p = Path(out_dir) / subject_id
    out_dir_p.mkdir(parents=True, exist_ok=True)

    try:
        if cfg.method == "precomputed":
            flair_brain, t1_brain, mask = _resolve_precomputed(
                flair_path, t1_path, out_dir_p
            )
        elif cfg.method == "synthstrip":
            flair_brain, t1_brain, mask = _run_synthstrip(
                flair_path, t1_path, out_dir_p, cfg
            )
        elif cfg.method == "hdbet":
            flair_brain, t1_brain, mask = _run_hdbet(
                flair_path, t1_path, out_dir_p, cfg
            )
        else:
            raise ValueError(f"Unknown stripping method: {cfg.method}")

        return StripResult(
            subject_id=subject_id,
            flair_brain_path=str(flair_brain),
            t1_brain_path=str(t1_brain),
            brain_mask_path=str(mask),
            seconds=time.perf_counter() - t0,
            success=True,
        )
    except Exception as exc:
        logger.exception("Skull stripping failed for %s", subject_id)
        return StripResult(
            subject_id=subject_id,
            flair_brain_path="",
            t1_brain_path="",
            brain_mask_path="",
            seconds=time.perf_counter() - t0,
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _resolve_precomputed(flair_path, t1_path, out_dir) -> tuple[Path, Path, Path]:
    """Look for already-stripped files next to the originals."""
    flair_brain = Path(str(flair_path).replace(".nii.gz", "_brain.nii.gz"))
    t1_brain = Path(str(t1_path).replace(".nii.gz", "_brain.nii.gz"))
    mask = Path(str(t1_path).replace(".nii.gz", "_brain_mask.nii.gz"))
    for p in (flair_brain, t1_brain, mask):
        if not p.exists():
            raise FileNotFoundError(f"Precomputed strip not found: {p}")
    return flair_brain, t1_brain, mask


def _run_synthstrip(flair_path, t1_path, out_dir, cfg) -> tuple[Path, Path, Path]:
    """Run FreeSurfer SynthStrip via Singularity.

    SynthStrip is fast on CPU (~30 s per volume) and works well for FLAIR.
    We run it on T1 to get a mask, then apply that mask to FLAIR after
    coregistration in the next stage. Here we strip both independently to
    stay simple; registration is the next stage.
    """
    sif = cfg.container_path or "synthstrip.sif"
    if not Path(sif).exists() and not _is_command("singularity"):
        # Soft fallback for laptop dev: use a no-op strip (assume already brain).
        logger.warning("SynthStrip not available; copying inputs through.")
        flair_brain = out_dir / "flair_brain.nii.gz"
        t1_brain = out_dir / "t1_brain.nii.gz"
        mask = out_dir / "brain_mask.nii.gz"
        shutil.copyfile(flair_path, flair_brain)
        shutil.copyfile(t1_path, t1_brain)
        shutil.copyfile(t1_path, mask)  # placeholder
        return flair_brain, t1_brain, mask

    flair_brain = out_dir / "flair_brain.nii.gz"
    t1_brain = out_dir / "t1_brain.nii.gz"
    mask = out_dir / "brain_mask.nii.gz"

    # T1 with mask output.
    _run([
        "singularity", "exec", sif,
        "mri_synthstrip", "-i", str(t1_path),
        "-o", str(t1_brain), "-m", str(mask),
    ])
    # FLAIR using the same brain extraction (independent strip; registration
    # later aligns them precisely).
    _run([
        "singularity", "exec", sif,
        "mri_synthstrip", "-i", str(flair_path),
        "-o", str(flair_brain),
    ])
    return flair_brain, t1_brain, mask


def _run_hdbet(flair_path, t1_path, out_dir, cfg) -> tuple[Path, Path, Path]:
    """Run HD-BET via Singularity. GPU strongly recommended."""
    sif = cfg.container_path or "hdbet.sif"
    device = "0" if cfg.gpu else "cpu"

    flair_brain = out_dir / "flair_brain.nii.gz"
    t1_brain = out_dir / "t1_brain.nii.gz"
    mask = out_dir / "t1_brain_mask.nii.gz"

    _run([
        "singularity", "exec", "--nv" if cfg.gpu else "exec", sif,
        "hd-bet", "-i", str(t1_path),
        "-o", str(t1_brain), "-device", device, "-mode", "fast",
    ])
    _run([
        "singularity", "exec", "--nv" if cfg.gpu else "exec", sif,
        "hd-bet", "-i", str(flair_path),
        "-o", str(flair_brain), "-device", device, "-mode", "fast",
    ])
    return flair_brain, t1_brain, mask


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> None:
    """Run a shell command, raising on non-zero exit."""
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
        )


def _is_command(name: str) -> bool:
    return shutil.which(name) is not None
