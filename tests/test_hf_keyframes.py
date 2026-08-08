from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aic.hf_keyframes import (
    batches,
    build_dataset_index,
    build_readme,
    collect_upload_entries,
    official_video_pairs,
)


def _write_pair(root: Path) -> tuple[str, str]:
    archive = "Videos_L21_a"
    video_id = "L21_V001"
    stem = f"{archive}--{video_id}"
    tar_name = f"{stem}-keyframes.tar"
    report_name = f"{stem}-report.json"
    tar_path = root / tar_name
    tar_path.write_bytes(b"tar bytes")
    report = {
        "status": "done",
        "archive": archive,
        "video_id": video_id,
        "artifact": tar_name,
        "artifact_bytes": tar_path.stat().st_size,
        "artifact_sha256": hashlib.sha256(tar_path.read_bytes()).hexdigest(),
        "videos": 1,
        "shots": 3,
        "keyframes": 7,
    }
    (root / report_name).write_text(json.dumps(report), encoding="utf-8")
    return archive, video_id


def test_collect_upload_entries_preserves_verified_names(tmp_path: Path) -> None:
    pair = _write_pair(tmp_path)
    entries, reports = collect_upload_entries(tmp_path, [pair])
    assert [entry.kind for entry in entries] == ["tar", "report"]
    assert entries[0].path_in_repo == (
        "packages/Videos_L21_a/"
        "Videos_L21_a--L21_V001-keyframes.tar"
    )
    assert entries[0].size == len(b"tar bytes")
    assert reports[0]["keyframes"] == 7


def test_collect_upload_entries_rejects_size_mismatch(tmp_path: Path) -> None:
    pair = _write_pair(tmp_path)
    report_path = next(tmp_path.glob("*-report.json"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["artifact_bytes"] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_bytes"):
        collect_upload_entries(tmp_path, [pair])


def test_index_readme_and_batching(tmp_path: Path) -> None:
    pair = _write_pair(tmp_path)
    entries, reports = collect_upload_entries(tmp_path, [pair])
    index_row = json.loads(build_dataset_index(entries, reports))
    assert index_row["video_id"] == "L21_V001"
    assert index_row["tar_path"] == entries[0].path_in_repo
    readme = build_readme(reports).decode("utf-8")
    assert "1 videos, 3" in readme
    assert "7 keyframes" in readme
    assert batches(entries, size=1) == [[entries[0]], [entries[1]]]
    with pytest.raises(ValueError, match="positive"):
        batches(entries, size=0)


def test_official_upload_scope_is_415_videos() -> None:
    assert len(official_video_pairs()) == 415
