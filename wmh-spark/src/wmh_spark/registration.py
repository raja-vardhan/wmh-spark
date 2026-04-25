"""MNI152 registration stage.

We use ANTsPy (antspyx) for affine + optional non-linear registration to the
MNI152 1mm template. Doing this once per subject lets all spatial-prior
features (distance-to-ventricle, lobe atlas, MNI coordinates) be computed in
a single common space.

Like skull stripping, this stage is embarrassingly parallel across subjects
and runs as a Spark ``rdd.map``.

Design notes:
- We register T1 -> MNI, then apply the resulting transform to the FLAIR
  *after* coregistering FLAIR -> T1 in subject space. This is the standard
  WMH preprocessing chain (e.g., Wardlaw 2013).
- We cache the warps so re-runs of feature extraction don't re-register.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import RegistrationConfig

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    subject_id: str
    flair_mni_path: str
    t1_mni_path: str
    brain_mask_mni_path: str
    transform_path: str
    seconds: float
    success: bool
    error: Optional[str] = None


def register_subject(
    subject_id: str,
    flair_brain_path: str,
    t1_brain_path: str,
    brain_mask_path: str,
    out_dir: str,
    cfg: RegistrationConfig,
) -> RegistrationResult:
    """Coregister FLAIR->T1, then warp T1+FLAIR+mask to MNI152.

    Wrapped in try/except so a single bad volume doesn't tear down the job.
    """
    t0 = time.perf_counter()
    out_dir_p = Path(out_dir) / subject_id
    out_dir_p.mkdir(parents=True, exist_ok=True)

    try:
        # Local import so workers without ANTs installed can still import the
        # module (useful for unit tests that don't exercise this stage).
        import ants

        # Step 1: FLAIR -> T1 in subject space (rigid, fast).
        t1_img = ants.image_read(t1_brain_path)
        flair_img = ants.image_read(flair_brain_path)
        coreg = ants.registration(
            fixed=t1_img, moving=flair_img, type_of_transform="Rigid"
        )
        flair_in_t1 = coreg["warpedmovout"]

        # Step 2: T1 -> MNI152.
        if not Path(cfg.template_path).exists():
            raise FileNotFoundError(
                f"MNI template missing at {cfg.template_path}. "
                "Download via fetch_templates.py."
            )
        template = ants.image_read(cfg.template_path)
        reg = ants.registration(
            fixed=template, moving=t1_img, type_of_transform=cfg.transform_type
        )

        # Apply the same warp to the (now T1-aligned) FLAIR and mask.
        flair_mni = ants.apply_transforms(
            fixed=template, moving=flair_in_t1,
            transformlist=reg["fwdtransforms"], interpolator="linear",
        )
        mask_img = ants.image_read(brain_mask_path)
        mask_mni = ants.apply_transforms(
            fixed=template, moving=mask_img,
            transformlist=reg["fwdtransforms"], interpolator="nearestNeighbor",
        )

        # Persist outputs.
        flair_out = out_dir_p / "flair_mni.nii.gz"
        t1_out = out_dir_p / "t1_mni.nii.gz"
        mask_out = out_dir_p / "brain_mask_mni.nii.gz"
        ants.image_write(flair_mni, str(flair_out))
        ants.image_write(reg["warpedmovout"], str(t1_out))
        ants.image_write(mask_mni, str(mask_out))

        # Cache the forward transform path. ANTs writes one or more files;
        # we record the first as a representative pointer.
        transform_path = reg["fwdtransforms"][0] if reg["fwdtransforms"] else ""

        return RegistrationResult(
            subject_id=subject_id,
            flair_mni_path=str(flair_out),
            t1_mni_path=str(t1_out),
            brain_mask_mni_path=str(mask_out),
            transform_path=transform_path,
            seconds=time.perf_counter() - t0,
            success=True,
        )
    except Exception as exc:
        logger.exception("Registration failed for %s", subject_id)
        return RegistrationResult(
            subject_id=subject_id,
            flair_mni_path="",
            t1_mni_path="",
            brain_mask_mni_path="",
            transform_path="",
            seconds=time.perf_counter() - t0,
            success=False,
            error=str(exc),
        )
