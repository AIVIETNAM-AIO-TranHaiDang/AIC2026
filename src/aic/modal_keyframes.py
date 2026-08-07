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


def _yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return raw


def make_runtime_config(modal_profile: Path, destination: Path) -> dict[str, Any]:
    """Apply the constrained Modal overlay to its declared base profile.

    Only cloud storage paths and the OmniShotCut selector may be overridden.
    Rejecting every other key prevents a hardware profile from silently
    changing keyframe, deduplication, embedding, or Chronicle behaviour.
    """
    import yaml  # noqa: PLC0415

    overlay = _yaml_mapping(modal_profile)
    allowed_top = {"base_profile", "paths", "ingest"}
    unexpected_top = set(overlay) - allowed_top
    if unexpected_top:
        raise ValueError(f"unsupported Modal override sections: {unexpected_top}")

    base_name = overlay.get("base_profile")
    if not isinstance(base_name, str) or Path(base_name).name != base_name:
        raise ValueError("base_profile must be a filename beside the Modal profile")
    source_profile = modal_profile.parent / base_name
    raw = _yaml_mapping(source_profile)

    paths = overlay.get("paths")
    ingest = overlay.get("ingest")
    if not isinstance(paths, dict) or set(paths) != {"data_root", "models_dir"}:
        raise ValueError("Modal paths override must contain data_root and models_dir")
    if not isinstance(ingest, dict) or set(ingest) != {"shots"}:
        raise ValueError("Modal ingest override may contain only shots")
    shots = ingest["shots"]
    allowed_shots = {"model", "checkpoint", "mode"}
    if not isinstance(shots, dict) or set(shots) != allowed_shots:
        raise ValueError(
            "Modal shots override must contain only model, checkpoint, and mode"
        )
    if shots["model"] != "omnishotcut":
        raise ValueError("Modal keyframe job requires shots.model: omnishotcut")

    raw_paths = raw.get("paths")
    raw_ingest = raw.get("ingest")
    if not isinstance(raw_paths, dict) or not isinstance(raw_ingest, dict):
        raise ValueError(f"invalid base profile: {source_profile}")
    raw_shots = raw_ingest.get("shots")
    if not isinstance(raw_shots, dict):
        raise ValueError(f"base profile has no ingest.shots: {source_profile}")
    raw_paths.update(paths)
    raw_shots.update(shots)

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


def parse_video_ids(archive_name: str, raw_video_ids: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI selection in deterministic archive order."""
    spec = get_archive_spec(archive_name)
    requested = {
        item.strip() for item in raw_video_ids.split(",") if item.strip()
    }
    if not requested:
        return spec.video_ids

    unknown = sorted(requested - set(spec.video_ids))
    if unknown:
        raise ValueError(
            f"videos do not belong to {archive_name}: {', '.join(unknown)}"
        )
    return tuple(video_id for video_id in spec.video_ids if video_id in requested)


def validate_ready_archive(input_root: Path, archive_name: str) -> dict[str, Any]:
    """Validate the completion marker for one persistent source ZIP.

    The marker is written only after the ZIP central directory and official
    video list have been checked.  A partial ZIP can therefore remain on the
    Volume for a resumed download without ever being consumed by a GPU job.
    """
    spec = get_archive_spec(archive_name)
    zip_path = input_root / f"{archive_name}.zip"
    marker_path = input_root / f"{archive_name}.ready.json"
    if not marker_path.is_file():
        raise ValueError(f"archive has not been staged: {marker_path.name}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid archive marker: {marker_path}") from exc
    if not isinstance(marker, dict):
        raise ValueError(f"invalid archive marker: {marker_path}")
    if marker.get("archive") != archive_name or marker.get("url") != spec.url:
        raise ValueError(f"archive marker does not match {archive_name}")
    if marker.get("videos") != len(spec.video_ids):
        raise ValueError(f"archive marker has the wrong video count: {marker_path}")
    if not zip_path.is_file() or zip_path.stat().st_size <= 0:
        raise ValueError(f"staged ZIP is missing or empty: {zip_path}")
    if marker.get("artifact_bytes") != zip_path.stat().st_size:
        raise ValueError(f"staged ZIP size does not match its marker: {zip_path}")
    checksum = marker.get("artifact_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError(f"archive marker has no SHA-256: {marker_path}")
    return marker


def completed_video_result(
    results_root: Path, archive_name: str, video_id: str
) -> dict[str, Any] | None:
    """Return a completed per-video report, or ``None`` if it must rerun.

    Reports produced by the original one-video pilot did not yet contain a
    ``status`` field.  They are accepted when every other completion invariant
    matches so that the verified pilot is not recomputed.
    """
    stem = output_stem(archive_name, video_id)
    artifact_path = results_root / f"{stem}-keyframes.tar"
    report_path = results_root / f"{stem}-report.json"
    if not artifact_path.is_file() or not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(report, dict):
        return None
    expected = {
        "archive": archive_name,
        "video_id": video_id,
        "artifact": artifact_path.name,
        "artifact_bytes": artifact_path.stat().st_size,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        return None
    if report.get("status", "done") != "done":
        return None
    checksum = report.get("artifact_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        return None
    return report


def sha256_file(path: Path) -> str:
    """Compute a download-verification hash without loading a tar into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
