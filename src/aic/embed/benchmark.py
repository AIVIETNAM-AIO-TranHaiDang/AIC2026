"""Deterministic sample preparation and validation for embedding benchmarks."""

from __future__ import annotations

import math
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PACKAGE_RE = re.compile(
    r"^Videos_L\d{2}_[ab]--L\d{2}_V\d{3}-keyframes\.tar$"
)
_KEYFRAME_PREFIX = "data/keyframes/"


@dataclass(frozen=True)
class SampleImage:
    """One JPEG copied from a verified per-video keyframe package."""

    source_package: str
    source_member: str
    local_path: Path

    @property
    def keyframe_id(self) -> str:
        return Path(self.source_member).stem


def keyframe_packages(results_root: Path) -> list[Path]:
    """Return deterministic, report-backed per-video keyframe TARs."""
    packages: list[Path] = []
    for path in sorted(results_root.glob("*-keyframes.tar")):
        if not _PACKAGE_RE.fullmatch(path.name):
            continue
        report = path.with_name(
            path.name.removesuffix("-keyframes.tar") + "-report.json"
        )
        if report.is_file():
            packages.append(path)
    return packages


def extract_keyframe_sample(
    results_root: Path,
    destination: Path,
    sample_size: int,
) -> list[SampleImage]:
    """Copy exactly ``sample_size`` JPEGs from sorted verified packages.

    Members are streamed explicitly instead of using ``extractall`` so paths
    from an archive can never escape ``destination``.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError(f"sample destination must be empty: {destination}")

    records: list[SampleImage] = []
    for package in keyframe_packages(results_root):
        with tarfile.open(package, mode="r") as bundle:
            members = sorted(
                (
                    member
                    for member in bundle.getmembers()
                    if member.isfile()
                    and member.name.startswith(_KEYFRAME_PREFIX)
                    and member.name.lower().endswith(".jpg")
                ),
                key=lambda member: member.name,
            )
            for member in members:
                source = bundle.extractfile(member)
                if source is None:
                    raise RuntimeError(
                        f"cannot read {member.name} from {package.name}"
                    )
                local_path = destination / f"sample-{len(records):05d}.jpg"
                with source, local_path.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if local_path.stat().st_size == 0:
                    raise RuntimeError(
                        f"empty JPEG {member.name} in {package.name}"
                    )
                records.append(
                    SampleImage(
                        source_package=package.name,
                        source_member=member.name,
                        local_path=local_path,
                    )
                )
                if len(records) == sample_size:
                    return records

    raise ValueError(
        f"requested {sample_size} images but only found {len(records)} "
        f"in {results_root}"
    )


def vector_health(vectors: np.ndarray, expected_rows: int) -> dict[str, float | int]:
    """Validate source encoder invariants and return compact health metrics."""
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise ValueError(
            f"expected [{expected_rows}, dim] vectors, got {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("benchmark vectors contain NaN or Inf")
    norms = np.linalg.norm(vectors, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    if not math.isfinite(max_norm_error) or max_norm_error > 1e-4:
        raise ValueError(
            f"vectors are not L2-normalised; max error={max_norm_error}"
        )
    return {
        "rows": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "max_norm_error": max_norm_error,
    }


def estimated_compute_cost(
    elapsed_seconds: float,
    *,
    gpu_usd_per_second: float,
    cpu_cores: float,
    memory_gib: float,
    cpu_usd_per_core_second: float = 0.0000131,
    memory_usd_per_gib_second: float = 0.00000222,
) -> float:
    """Estimate Modal compute cost using explicit, reviewable resource rates."""
    rate = (
        gpu_usd_per_second
        + cpu_cores * cpu_usd_per_core_second
        + memory_gib * memory_usd_per_gib_second
    )
    return elapsed_seconds * rate
