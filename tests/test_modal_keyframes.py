from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aic.config import load_config
from aic.modal_keyframes import (
    ARCHIVE_SPECS,
    completed_video_result,
    get_archive_spec,
    make_runtime_config,
    output_stem,
    parse_video_ids,
    select_zip_members,
    validate_ingest_output,
    validate_ready_archive,
)


def test_official_archive_scope_is_415_videos() -> None:
    assert list(ARCHIVE_SPECS) == [
        "Videos_L21_a",
        "Videos_L22_a",
        "Videos_L23_a",
        "Videos_L24_a",
        "Videos_L25_a",
        "Videos_L26_a",
        "Videos_L26_b",
    ]
    assert sum(len(spec.video_ids) for spec in ARCHIVE_SPECS.values()) == 415
    assert "L21_V004" not in ARCHIVE_SPECS["Videos_L21_a"].video_ids
    assert ARCHIVE_SPECS["Videos_L26_b"].video_ids[0] == "L26_V100"


def test_select_single_nested_video_member() -> None:
    spec = get_archive_spec("Videos_L21_a")
    selected = select_zip_members(
        ["Videos_L21_a/readme.txt", "Videos_L21_a/L21_V001.mp4"],
        spec,
        "L21_V001",
    )
    assert selected == [("Videos_L21_a/L21_V001.mp4", "L21_V001")]


def test_select_rejects_video_outside_archive() -> None:
    with pytest.raises(ValueError, match="does not belong"):
        select_zip_members([], get_archive_spec("Videos_L21_a"), "L22_V001")


def test_runtime_config_only_overrides_modal_fields(tmp_path: Path) -> None:
    source = Path("configs/t0.yaml")
    overlay = Path("configs/modal-keyframes.yaml")
    destination = tmp_path / "modal.yaml"
    generated = make_runtime_config(overlay, destination)
    original = yaml.safe_load(source.read_text(encoding="utf-8"))

    expected = yaml.safe_load(source.read_text(encoding="utf-8"))
    expected["paths"].update({"data_root": "data", "models_dir": "/models"})
    expected["ingest"]["shots"].update(
        {
            "model": "omnishotcut",
            "checkpoint": "uva-cv-lab/OmniShotCut",
            "mode": "default",
        }
    )
    assert generated == expected
    assert generated["project"] == original["project"]
    assert generated["ingest"]["keyframes"] == original["ingest"]["keyframes"]
    assert generated["ingest"]["dedup"] == original["ingest"]["dedup"]
    assert generated["paths"]["data_root"] == "data"
    assert generated["paths"]["models_dir"] == "/models"
    parsed = load_config(destination)
    assert parsed.ingest.keyframes == load_config(source).ingest.keyframes


def test_runtime_config_rejects_unrelated_override(tmp_path: Path) -> None:
    (tmp_path / "t0.yaml").write_text(
        Path("configs/t0.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    overlay = tmp_path / "modal.yaml"
    overlay.write_text(
        "base_profile: t0.yaml\n"
        "paths: {data_root: data, models_dir: /models}\n"
        "ingest:\n"
        "  shots: {model: omnishotcut, checkpoint: x, mode: default}\n"
        "embed: {batch_size: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported Modal override"):
        make_runtime_config(overlay, tmp_path / "generated.yaml")


def test_validate_source_output_shape(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    manifests = data_root / "manifests"
    image = data_root / "keyframes" / "L21_V001" / "L21_V001_s0_f1.jpg"
    manifests.mkdir(parents=True)
    image.parent.mkdir(parents=True)
    image.write_bytes(b"jpg")

    rows = {
        "videos.jsonl": [{"video_id": "L21_V001", "status": "done"}],
        "shots.jsonl": [{"video_id": "L21_V001", "shot_id": 0}],
        "keyframes.jsonl": [
            {
                "keyframe_id": "L21_V001_s0_f1",
                "video_id": "L21_V001",
                "image_path": "data/keyframes/L21_V001/L21_V001_s0_f1.jpg",
            }
        ],
    }
    for name, records in rows.items():
        (manifests / name).write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    assert validate_ingest_output(data_root, {"L21_V001"}) == {
        "videos": 1,
        "shots": 1,
        "keyframes": 1,
    }


def test_pilot_and_full_outputs_cannot_collide() -> None:
    assert output_stem("Videos_L21_a", "L21_V001") == ("Videos_L21_a--L21_V001")
    assert output_stem("Videos_L21_a", None) == "Videos_L21_a"


def test_parse_video_ids_defaults_to_archive_and_restores_official_order() -> None:
    expected = ARCHIVE_SPECS["Videos_L21_a"].video_ids
    assert parse_video_ids("Videos_L21_a", "") == expected
    assert parse_video_ids("Videos_L21_a", "L21_V003, L21_V001,L21_V003") == (
        "L21_V001",
        "L21_V003",
    )
    with pytest.raises(ValueError, match="do not belong"):
        parse_video_ids("Videos_L21_a", "L22_V001")


def test_ready_archive_requires_matching_marker_and_zip(tmp_path: Path) -> None:
    archive = "Videos_L21_a"
    zip_path = tmp_path / f"{archive}.zip"
    zip_path.write_bytes(b"official zip placeholder")
    marker = {
        "status": "ready",
        "archive": archive,
        "url": ARCHIVE_SPECS[archive].url,
        "videos": len(ARCHIVE_SPECS[archive].video_ids),
        "artifact": zip_path.name,
        "artifact_bytes": zip_path.stat().st_size,
        "artifact_sha256": "a" * 64,
    }
    (tmp_path / f"{archive}.ready.json").write_text(
        json.dumps(marker), encoding="utf-8"
    )
    assert validate_ready_archive(tmp_path, archive) == marker

    zip_path.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="size does not match"):
        validate_ready_archive(tmp_path, archive)


def test_completed_video_result_accepts_verified_legacy_pilot(tmp_path: Path) -> None:
    archive = "Videos_L21_a"
    video_id = "L21_V001"
    stem = output_stem(archive, video_id)
    artifact = tmp_path / f"{stem}-keyframes.tar"
    artifact.write_bytes(b"tar output")
    report = {
        "archive": archive,
        "video_id": video_id,
        "artifact": artifact.name,
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": "b" * 64,
    }
    report_path = tmp_path / f"{stem}-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert completed_video_result(tmp_path, archive, video_id) == report

    report["artifact_bytes"] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert completed_video_result(tmp_path, archive, video_id) is None
