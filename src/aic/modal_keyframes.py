"""Pure helpers for running the official ZIP archives on Modal.

This module deliberately contains no Modal imports.  Archive validation,
runtime-config generation, and output validation can therefore be tested on
the development machine without installing Modal into the project venv.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


@dataclass(frozen=True)
class ArchiveSpec:
    """One official input ZIP and the video IDs it must contain."""

    name: str
    url: str
    level: int
    first_video: int
    last_video: int
    excluded_videos: frozenset[int] = frozenset()

    @property
    def video_ids(self) -> tuple[str, ...]:
        return tuple(
            f"L{self.level:02d}_V{number:03d}"
            for number in range(self.first_video, self.last_video + 1)
            if number not in self.excluded_videos
        )


ARCHIVE_SPECS: dict[str, ArchiveSpec] = {
    spec.name: spec
    for spec in (
        ArchiveSpec(
            "Videos_L21_a",
            "https://aic-data.ledo.io.vn/Videos_L21_a.zip",
            21,
            1,
            31,
            frozenset({4, 20}),
        ),
        ArchiveSpec(
            "Videos_L22_a",
            "https://aic-data.ledo.io.vn/Videos_L22_a.zip",
            22,
            1,
            31,
        ),
        ArchiveSpec(
            "Videos_L23_a",
            "https://aic-data.ledo.io.vn/Videos_L23_a.zip",
            23,
            1,
            25,
        ),
        ArchiveSpec(
            "Videos_L24_a",
            "https://aic-data.ledo.io.vn/Videos_L24_a.zip",
            24,
            2,
            45,
            frozenset({34}),
        ),
        ArchiveSpec(
            "Videos_L25_a",
            "https://aic-data.ledo.io.vn/Videos_L25_a.zip",
            25,
            1,
            88,
        ),
        ArchiveSpec(
            "Videos_L26_a",
            "https://aic-data.ledo.io.vn/Videos_L26_a.zip",
            26,
            1,
            99,
        ),
        ArchiveSpec(
            "Videos_L26_b",
            "https://aic-data.ledo.io.vn/Videos_L26_b.zip",
            26,
            100,
            199,
        ),
    )
}


def get_archive_spec(name: str) -> ArchiveSpec:
    """Return a supported archive or fail before any cloud work starts."""
    try:
        return ARCHIVE_SPECS[name]
    except KeyError as exc:
        choices = ", ".join(ARCHIVE_SPECS)
        raise ValueError(f"unknown archive {name!r}; choose one of: {choices}") from exc


def select_zip_members(
    member_names: list[str], spec: ArchiveSpec, video_id: str | None = None
) -> list[tuple[str, str]]:
    """Map safe ZIP members to ``(member_name, video_id)`` pairs.

    Members may be nested in a directory inside the ZIP.  Extraction code uses
    only the validated basename, so an archive cannot write outside the work
    directory.  A whole-archive run also checks that every video expected from
    the official sheet is present exactly once.
    """
    expected = set(spec.video_ids)
    if video_id is not None and video_id not in expected:
        raise ValueError(f"{video_id!r} does not belong to {spec.name}")
    wanted = {video_id} if video_id else expected
    selected: dict[str, str] = {}

    for raw_name in member_names:
        normalised = raw_name.replace("\\", "/")
        path = PurePosixPath(normalised)
        if path.suffix.lower() != ".mp4" or path.stem not in wanted:
            continue
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe ZIP member path: {raw_name!r}")
        if path.stem in selected:
            raise ValueError(f"duplicate video {path.stem!r} in {spec.name}")
        selected[path.stem] = raw_name

    missing = sorted(wanted - selected.keys())
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"{spec.name} is missing {preview}{suffix}")
    return [(selected[item], item) for item in sorted(selected)]


def make_runtime_config(
    source_profile: Path, destination: Path, *, models_dir: str = "/models"
) -> dict[str, Any]:
    """Clone the checked-in profile, changing storage paths and nothing else."""
    raw = yaml.safe_load(source_profile.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("paths"), dict):
        raise ValueError(f"invalid source profile: {source_profile}")
    raw["paths"]["data_root"] = "data"
    raw["paths"]["models_dir"] = models_dir
    destination.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return raw


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing output manifest: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path} line {line_no}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"non-object record in {path} line {line_no}")
            records.append(record)
    return records


def validate_ingest_output(data_root: Path, expected_ids: set[str]) -> dict[str, int]:
    """Validate the three source manifests and every referenced JPG."""
    manifests = data_root / "manifests"
    video_rows = _read_jsonl(manifests / "videos.jsonl")
    shot_rows = _read_jsonl(manifests / "shots.jsonl")
    keyframe_rows = _read_jsonl(manifests / "keyframes.jsonl")

    done_ids = {
        str(row.get("video_id")) for row in video_rows if row.get("status") == "done"
    }
    failed_ids = {
        str(row.get("video_id")) for row in video_rows if row.get("status") != "done"
    }
    if done_ids != expected_ids or failed_ids:
        raise ValueError(
            "ingest did not finish the requested videos: "
            f"done={sorted(done_ids)}, failed={sorted(failed_ids)}, "
            f"expected={sorted(expected_ids)}"
        )

    shot_video_ids = {str(row.get("video_id")) for row in shot_rows}
    keyframe_video_ids = {str(row.get("video_id")) for row in keyframe_rows}
    if not expected_ids.issubset(shot_video_ids):
        raise ValueError("one or more completed videos have no shot records")
    if not expected_ids.issubset(keyframe_video_ids):
        raise ValueError("one or more completed videos have no keyframe records")

    work_root = data_root.parent
    for row in keyframe_rows:
        raw_path = row.get("image_path")
        if not isinstance(raw_path, str):
            raise ValueError("keyframe manifest row has no image_path")
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = work_root / image_path
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise ValueError(f"missing or empty keyframe image: {image_path}")

    return {
        "videos": len(done_ids),
        "shots": len(shot_rows),
        "keyframes": len(keyframe_rows),
    }


def output_stem(archive_name: str, video_id: str | None) -> str:
    """Give pilot and full-archive artifacts distinct deterministic names."""
    spec = get_archive_spec(archive_name)
    if video_id is None:
        return archive_name
    if video_id not in spec.video_ids:
        raise ValueError(f"{video_id!r} does not belong to {archive_name}")
    return f"{archive_name}--{video_id}"


def sha256_file(path: Path) -> str:
    """Compute a download-verification hash without loading a tar into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
