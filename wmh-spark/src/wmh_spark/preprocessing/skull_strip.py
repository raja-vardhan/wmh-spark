"""HD-BET skull-stripping preprocessing module.

Orchestrates Docker-based HD-BET execution on raw T1/FLAIR NIfTI scan pairs,
applies a quality gate, and forwards accepted outputs to the ingestion stage.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DockerExecutionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Data classes  (T004)
# ---------------------------------------------------------------------------


@dataclass
class RawScan:
    path: Path
    modality: Literal["T1", "FLAIR"]
    subject_id: str


@dataclass
class ScanPair:
    subject_id: str
    t1_scan: RawScan
    flair_scan: RawScan
    t1_brain_ref_path: Optional[Path] = None
    flair_brain_ref_path: Optional[Path] = None


@dataclass
class SkullStrippedOutput:
    subject_id: str
    t1_stripped_path: Optional[Path] = None
    t1_mask_path: Optional[Path] = None
    flair_stripped_path: Optional[Path] = None
    flair_mask_path: Optional[Path] = None
    t1_dsc: Optional[float] = None
    flair_dsc: Optional[float] = None
    status: Literal["accepted", "rejected", "error"] = "error"
    error_message: Optional[str] = None
    elapsed_seconds: float = 0.0


@dataclass
class ProcessingLogEntry:
    run_id: str
    timestamp: str
    subject_id: str
    t1_dsc: Optional[float]
    flair_dsc: Optional[float]
    status: str
    elapsed_seconds: float
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Docker runner  (T011)
# ---------------------------------------------------------------------------


class DockerRunner:
    def __init__(self, image: str, use_gpu: bool = False) -> None:
        self.image = image
        self.use_gpu = use_gpu

    def _build_command(self, input_path: Path, output_path: Path) -> List[str]:
        cmd: List[str] = ["docker", "run", "--rm"]
        if self.use_gpu:
            cmd.extend(["--gpus", "all"])
        cmd.extend(
            [
                "-v", f"{input_path.parent}:/input:ro",
                "-v", f"{output_path.parent}:/output",
                self.image,
                "-i", f"/input/{input_path.name}",
                "-o", f"/output/{output_path.name}",
            ]
        )
        return cmd

    def run(self, input_path: Path, output_path: Path) -> None:
        cmd = self._build_command(input_path, output_path)
        logger.debug("DockerRunner cmd: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise DockerExecutionError(
                f"HD-BET failed for {input_path.name}: {result.stderr.strip()}"
            )


# ---------------------------------------------------------------------------
# Pair discovery  (T012)
# ---------------------------------------------------------------------------


def _derive_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _discover_pairs(input_dir: Path) -> List[ScanPair]:
    """Scan input_dir for T1/FLAIR NIfTI pairs grouped by subdirectory.

    Each subdirectory is treated as one subject. Files in the root of
    input_dir are grouped as a single subject (subject_id = directory name).
    Partial pairs (T1 without FLAIR, or vice versa) are logged and skipped.
    """
    nifti_files = sorted(
        list(input_dir.rglob("*.nii.gz")) + list(input_dir.rglob("*.nii"))
    )

    by_dir: dict[Path, list[Path]] = defaultdict(list)
    for f in nifti_files:
        by_dir[f.parent].append(f)

    pairs: List[ScanPair] = []
    seen_ids: set[str] = set()

    for directory in sorted(by_dir):
        files = by_dir[directory]
        subject_id = directory.name

        t1_files = [f for f in files if re.search(r"T1", f.name)]
        flair_files = [f for f in files if re.search(r"FLAIR", f.name, re.IGNORECASE)]

        if not t1_files or not flair_files:
            logger.warning(
                "Partial pair in %s — T1=%d FLAIR=%d; skipping",
                directory, len(t1_files), len(flair_files),
            )
            continue

        if subject_id in seen_ids:
            raise ValueError(
                f"Duplicate subject_id '{subject_id}' found in input batch"
            )
        seen_ids.add(subject_id)

        pairs.append(ScanPair(
            subject_id=subject_id,
            t1_scan=RawScan(path=t1_files[0], modality="T1", subject_id=subject_id),
            flair_scan=RawScan(path=flair_files[0], modality="FLAIR", subject_id=subject_id),
        ))

    if not pairs:
        raise ValueError(f"No complete T1/FLAIR pairs found in {input_dir}")

    return pairs


# ---------------------------------------------------------------------------
# SkullStripper  (T013, T014, T015, T025, T026, T028)
# ---------------------------------------------------------------------------


class SkullStripper:
    DEFAULT_IMAGE = "wmh-spark/hdbet:2.0.0"

    def __init__(
        self,
        output_dir: Path | str,
        docker_image: str = DEFAULT_IMAGE,
        smoke_test: bool = False,
        benchmark_out: Optional[Path | str] = None,
        use_gpu: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.docker_image = docker_image
        self.smoke_test = smoke_test
        self.benchmark_out = Path(benchmark_out) if benchmark_out else None
        self._runner = DockerRunner(docker_image, use_gpu=use_gpu)
        self._run_id = str(uuid.uuid4())[:8]

    def process_pair(self, pair: ScanPair) -> SkullStrippedOutput:
        """Run HD-BET on one ScanPair and apply the quality gate.  (T013)"""
        from wmh_spark.preprocessing.quality_gate import QualityGate, QualityGateError

        t0 = time.time()
        subject_dir = self.output_dir / pair.subject_id

        # Clear existing outputs for deterministic overwrite  (T026)
        if subject_dir.exists():
            shutil.rmtree(subject_dir)
        subject_dir.mkdir(parents=True, exist_ok=True)

        t1_stem = _derive_stem(pair.t1_scan.path)
        flair_stem = _derive_stem(pair.flair_scan.path)

        t1_out = subject_dir / f"{t1_stem}_bet.nii.gz"
        t1_mask_out = subject_dir / f"{t1_stem}_bet_mask.nii.gz"
        flair_out = subject_dir / f"{flair_stem}_bet.nii.gz"
        flair_mask_out = subject_dir / f"{flair_stem}_bet_mask.nii.gz"

        # Docker execution
        try:
            self._runner.run(pair.t1_scan.path, t1_out)
            self._runner.run(pair.flair_scan.path, flair_out)
        except DockerExecutionError as exc:
            return SkullStrippedOutput(
                subject_id=pair.subject_id,
                status="error",
                error_message=str(exc),
                elapsed_seconds=time.time() - t0,
            )

        output = SkullStrippedOutput(
            subject_id=pair.subject_id,
            t1_stripped_path=t1_out,
            t1_mask_path=t1_mask_out,
            flair_stripped_path=flair_out,
            flair_mask_path=flair_mask_out,
        )

        # Quality gate  (T021)
        gate = QualityGate(smoke_test=self.smoke_test)
        try:
            if self.smoke_test:
                gate.assert_structural(t1_mask_out, pair.t1_scan.path)
                gate.assert_structural(flair_mask_out, pair.flair_scan.path)
            else:
                if pair.t1_brain_ref_path:
                    output.t1_dsc = gate.check_dsc(t1_mask_out, pair.t1_brain_ref_path)
                if pair.flair_brain_ref_path:
                    output.flair_dsc = gate.check_dsc(flair_mask_out, pair.flair_brain_ref_path)
            output.status = "accepted"
        except (QualityGateError, Exception) as exc:
            # Quarantine both outputs atomically  (T022)
            quarantine_dir = self.output_dir / "quarantine" / pair.subject_id
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            for src in [t1_out, t1_mask_out, flair_out, flair_mask_out]:
                if src.exists():
                    src.rename(quarantine_dir / src.name)
            output.status = "rejected"
            output.error_message = str(exc)

        output.elapsed_seconds = time.time() - t0
        return output

    def process_batch(
        self,
        input_dir: Optional[Path | str] = None,
        pairs: Optional[List[ScanPair]] = None,
        subject: Optional[str] = None,
    ) -> List[SkullStrippedOutput]:
        """Process a batch of scan pairs.  (T014)

        Args:
            input_dir: Directory to discover pairs from (alternative to pairs).
            pairs: Explicit list of ScanPair objects.
            subject: If set, process only this subject_id  (T025).
        """
        if pairs is None:
            if input_dir is None:
                raise ValueError("Either input_dir or pairs must be provided")
            pairs = _discover_pairs(Path(input_dir))

        if subject is not None:
            pairs = [p for p in pairs if p.subject_id == subject]
            if not pairs:
                raise ValueError(f"Subject '{subject}' not found in input batch")

        results: List[SkullStrippedOutput] = []
        log_entries: List[ProcessingLogEntry] = []
        batch_t0 = time.time()

        for pair in pairs:
            result = self.process_pair(pair)
            results.append(result)
            log_entries.append(ProcessingLogEntry(
                run_id=self._run_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                subject_id=result.subject_id,
                t1_dsc=result.t1_dsc,
                flair_dsc=result.flair_dsc,
                status=result.status,
                elapsed_seconds=result.elapsed_seconds,
                error_message=result.error_message,
            ))

        self._write_log(log_entries, batch_elapsed=time.time() - batch_t0)
        return results

    def process_from_manifest(
        self,
        manifest_path: Path | str,
        subject: Optional[str] = None,
    ) -> List[SkullStrippedOutput]:
        """Build ScanPair list from a Parquet manifest and run process_batch.  (T028)"""
        import pandas as pd

        manifest = pd.read_parquet(str(manifest_path))
        pairs: List[ScanPair] = []
        for _, row in manifest.iterrows():
            pairs.append(ScanPair(
                subject_id=row["subject_id"],
                t1_scan=RawScan(
                    path=Path(row["t1_path"]),
                    modality="T1",
                    subject_id=row["subject_id"],
                ),
                flair_scan=RawScan(
                    path=Path(row["flair_path"]),
                    modality="FLAIR",
                    subject_id=row["subject_id"],
                ),
            ))
        return self.process_batch(pairs=pairs, subject=subject)

    def _write_log(
        self, entries: List[ProcessingLogEntry], batch_elapsed: float
    ) -> None:
        """Write per-pair JSONL log and batch benchmark JSON.  (T015)

        Benchmark fields include subjects_per_hour per Constitution Principle V.
        """
        log_dir = Path("data/output/metrics")
        log_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = log_dir / f"skull_strip_{self._run_id}.jsonl"
        with open(jsonl_path, "a") as f:
            for entry in entries:
                f.write(json.dumps(dataclasses.asdict(entry)) + "\n")

        total = len(entries)
        accepted = sum(1 for e in entries if e.status == "accepted")
        rejected = sum(1 for e in entries if e.status == "rejected")
        errors = sum(1 for e in entries if e.status == "error")
        subjects_per_hour = (total / batch_elapsed * 3600) if batch_elapsed > 0 else 0.0

        benchmark_path = self.benchmark_out or log_dir / "skull_strip_benchmark.json"
        with open(str(benchmark_path), "w") as f:
            json.dump({
                "run_id": self._run_id,
                "total_pairs": total,
                "accepted": accepted,
                "rejected": rejected,
                "errors": errors,
                "total_elapsed_seconds": round(batch_elapsed, 4),
                "subjects_per_hour": round(subjects_per_hour, 2),
            }, f, indent=2)

        logger.info(
            "skull_strip batch complete: run_id=%s total=%d accepted=%d "
            "rejected=%d errors=%d elapsed=%.2fs subjects/hr=%.1f",
            self._run_id, total, accepted, rejected, errors,
            batch_elapsed, subjects_per_hour,
        )


# ---------------------------------------------------------------------------
# CLI entry point  (T027)
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HD-BET skull-stripping on T1/FLAIR NIfTI scan pairs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=Path, help="Directory of raw NIfTI scans")
    group.add_argument("--manifest", type=Path, help="Parquet manifest (SubjectRecord schema)")

    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--subject", type=str, default=None, help="Process single subject only")
    parser.add_argument("--smoke-test", action="store_true", help="Phase 1 mode: skip DSC gate")
    parser.add_argument(
        "--docker-image", type=str, default=SkullStripper.DEFAULT_IMAGE,
        help="Docker image tag (default: wmh-spark/hdbet:2.0.0)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Pass --gpus all to docker run (use a CUDA-based image, not the CPU Dockerfile)",
    )
    parser.add_argument("--benchmark-out", type=Path, default=None, help="Benchmark JSON path")

    args = parser.parse_args()

    stripper = SkullStripper(
        output_dir=args.output_dir,
        docker_image=args.docker_image,
        smoke_test=args.smoke_test,
        benchmark_out=args.benchmark_out,
        use_gpu=args.use_gpu,
    )

    try:
        if args.manifest:
            results = stripper.process_from_manifest(args.manifest, subject=args.subject)
        else:
            results = stripper.process_batch(
                input_dir=args.input_dir, subject=args.subject
            )
    except Exception as exc:
        logger.error("Fatal error: %s", exc)
        return 2

    errors = [r for r in results if r.status == "error"]
    rejected = [r for r in results if r.status == "rejected"]

    if errors:
        return 2
    if rejected:
        return 1
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(_main())
