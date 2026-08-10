from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from aic.embed.benchmark import (
    estimated_compute_cost,
    extract_keyframe_sample,
    keyframe_packages,
    vector_health,
)


def _write_package(root: Path, video_id: str, image_count: int) -> Path:
    stem = f"Videos_L21_a--{video_id}"
    package = root / f"{stem}-keyframes.tar"
    with tarfile.open(package, mode="w") as bundle:
        for index in range(image_count):
            payload = f"jpeg-{video_id}-{index}".encode()
            info = tarfile.TarInfo(
                f"data/keyframes/{video_id}/{video_id}_s0_f{index}.jpg"
            )
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
        ignored = b"{}\n"
        info = tarfile.TarInfo("data/manifests/keyframes.jsonl")
        info.size = len(ignored)
        bundle.addfile(info, io.BytesIO(ignored))
    (root / f"{stem}-report.json").write_text("{}", encoding="utf-8")
    return package


def test_extract_keyframe_sample_is_exact_and_deterministic(tmp_path: Path) -> None:
    _write_package(tmp_path, "L21_V002", 2)
    first = _write_package(tmp_path, "L21_V001", 2)
    destination = tmp_path / "sample"

    records = extract_keyframe_sample(tmp_path, destination, 3)

    assert keyframe_packages(tmp_path)[0] == first
    assert [record.keyframe_id for record in records] == [
        "L21_V001_s0_f0",
        "L21_V001_s0_f1",
        "L21_V002_s0_f0",
    ]
    assert [record.local_path.read_bytes() for record in records] == [
        b"jpeg-L21_V001-0",
        b"jpeg-L21_V001-1",
        b"jpeg-L21_V002-0",
    ]


def test_extract_requires_enough_images_and_empty_destination(tmp_path: Path) -> None:
    _write_package(tmp_path, "L21_V001", 1)
    with pytest.raises(ValueError, match="only found 1"):
        extract_keyframe_sample(tmp_path, tmp_path / "too-many", 2)

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        extract_keyframe_sample(tmp_path, nonempty, 1)


def test_vector_health_accepts_normalised_rows_and_rejects_nan() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    assert vector_health(vectors, 2) == {
        "rows": 2,
        "dim": 2,
        "max_norm_error": 0.0,
    }
    vectors[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        vector_health(vectors, 2)


def test_estimated_compute_cost_includes_gpu_cpu_and_memory() -> None:
    cost = estimated_compute_cost(
        10.0,
        gpu_usd_per_second=1.0,
        cpu_cores=2.0,
        memory_gib=4.0,
        cpu_usd_per_core_second=0.1,
        memory_usd_per_gib_second=0.01,
    )
    assert cost == pytest.approx(12.4)
