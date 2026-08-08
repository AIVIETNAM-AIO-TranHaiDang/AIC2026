"""Validate and map Modal keyframe packages for Hugging Face storage."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aic.modal_keyframes import ARCHIVE_SPECS, output_stem


@dataclass(frozen=True)
class UploadEntry:
    """One immutable local artifact and its destination in a dataset repo."""

    archive: str
    video_id: str
    kind: str
    source_path: Path
    path_in_repo: str
    size: int
    sha256: str


def official_video_pairs() -> tuple[tuple[str, str], ...]:
    """Return all official archive/video pairs in spreadsheet order."""
    return tuple(
        (archive, video_id)
        for archive, spec in ARCHIVE_SPECS.items()
        for video_id in spec.video_ids
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid report: {path}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"report must contain an object: {path}")
    return report


def collect_upload_entries(
    results_root: Path,
    pairs: Iterable[tuple[str, str]] | None = None,
    *,
    require_exact_files: bool = True,
) -> tuple[list[UploadEntry], list[dict[str, Any]]]:
    """Validate completed TAR/report pairs and map them into archive folders."""
    selected = tuple(pairs) if pairs is not None else official_video_pairs()
    entries: list[UploadEntry] = []
    reports: list[dict[str, Any]] = []
    expected_names: set[str] = set()

    for archive, video_id in selected:
        spec = ARCHIVE_SPECS.get(archive)
        if spec is None or video_id not in spec.video_ids:
            raise ValueError(f"invalid official selection: {archive}/{video_id}")
        stem = output_stem(archive, video_id)
        tar_name = f"{stem}-keyframes.tar"
        report_name = f"{stem}-report.json"
        tar_path = results_root / tar_name
        report_path = results_root / report_name
        expected_names.update((tar_name, report_name))
        if not tar_path.is_file() or tar_path.stat().st_size <= 0:
            raise ValueError(f"missing or empty TAR: {tar_path}")
        if not report_path.is_file() or report_path.stat().st_size <= 0:
            raise ValueError(f"missing or empty report: {report_path}")

        report = _read_report(report_path)
        status = report.get("status", "done")
        expected_fields = {
            "archive": archive,
            "video_id": video_id,
            "artifact": tar_name,
            "artifact_bytes": tar_path.stat().st_size,
            "videos": 1,
        }
        if status != "done":
            raise ValueError(f"report is not complete: {report_path}")
        for key, expected_value in expected_fields.items():
            if report.get(key) != expected_value:
                raise ValueError(
                    f"{report_path}: {key}={report.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        for key in ("shots", "keyframes"):
            value = report.get(key)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{report_path}: invalid {key}={value!r}")
        tar_sha256 = report.get("artifact_sha256")
        if not isinstance(tar_sha256, str) or len(tar_sha256) != 64:
            raise ValueError(f"{report_path}: invalid artifact_sha256")
        try:
            int(tar_sha256, 16)
        except ValueError as exc:
            raise ValueError(f"{report_path}: non-hex artifact_sha256") from exc

        destination_root = f"packages/{archive}"
        entries.extend(
            (
                UploadEntry(
                    archive=archive,
                    video_id=video_id,
                    kind="tar",
                    source_path=tar_path,
                    path_in_repo=f"{destination_root}/{tar_name}",
                    size=tar_path.stat().st_size,
                    sha256=tar_sha256,
                ),
                UploadEntry(
                    archive=archive,
                    video_id=video_id,
                    kind="report",
                    source_path=report_path,
                    path_in_repo=f"{destination_root}/{report_name}",
                    size=report_path.stat().st_size,
                    sha256=_sha256(report_path),
                ),
            )
        )
        reports.append(report)

    if require_exact_files:
        actual_names = {path.name for path in results_root.iterdir() if path.is_file()}
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        if missing or unexpected:
            raise ValueError(
                f"result Volume does not match official scope: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
    return entries, reports


def build_dataset_index(
    entries: list[UploadEntry], reports: list[dict[str, Any]]
) -> bytes:
    """Create one JSONL catalogue row per video without rewriting manifests."""
    tar_paths = {
        (entry.archive, entry.video_id): entry.path_in_repo
        for entry in entries
        if entry.kind == "tar"
    }
    report_paths = {
        (entry.archive, entry.video_id): entry.path_in_repo
        for entry in entries
        if entry.kind == "report"
    }
    rows = []
    for report in reports:
        key = (str(report["archive"]), str(report["video_id"]))
        rows.append(
            {
                "archive": key[0],
                "video_id": key[1],
                "shots": report["shots"],
                "keyframes": report["keyframes"],
                "tar_path": tar_paths[key],
                "report_path": report_paths[key],
                "artifact_bytes": report["artifact_bytes"],
                "artifact_sha256": report["artifact_sha256"],
            }
        )
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def build_readme(reports: list[dict[str, Any]]) -> bytes:
    """Generate a conservative private-dataset card for the verified packages."""
    total_videos = len(reports)
    total_shots = sum(int(report["shots"]) for report in reports)
    total_keyframes = sum(int(report["keyframes"]) for report in reports)
    text = f"""---
pretty_name: AIC2026 OmniShotCut keyframe packages
---

# AIC2026 keyframe packages

Private working dataset containing the verified output of the repository's
OmniShotCut keyframe stage. It contains {total_videos} videos, {total_shots}
shots, and {total_keyframes} keyframes.

Each video is stored as one uncompressed TAR plus one JSON report under
`packages/<official ZIP>/`. TAR contents preserve the source layout:

```text
data/keyframes/<video_id>/*.jpg
data/manifests/videos.jsonl
data/manifests/shots.jsonl
data/manifests/keyframes.jsonl
```

`dataset-index.jsonl` records the TAR path, report path, counts, byte size, and
SHA-256 for every video. These packages are storage/checkpoint artifacts, not
WebDataset shards; extract and merge the source JSONL manifests before running
later pipeline stages that expect filesystem image paths.
"""
    return text.encode("utf-8")


def batches(entries: list[UploadEntry], size: int = 50) -> list[list[UploadEntry]]:
    """Split upload operations into deterministic, bounded commits."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [entries[start : start + size] for start in range(0, len(entries), size)]
