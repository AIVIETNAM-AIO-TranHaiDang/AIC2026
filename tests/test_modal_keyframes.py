from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from aic.config import load_config
from aic.modal_keyframes import (
    ARCHIVE_SPECS,
    get_archive_spec,
    make_runtime_config,
    output_stem,
    select_zip_members,
    validate_ingest_output,
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


def test_runtime_config_only_overrides_paths(tmp_path: Path) -> None:
    source = Path("configs/t0-gtx1650.yaml")
    destination = tmp_path / "modal.yaml"
    generated = make_runtime_config(source, destination)
    original = yaml.safe_load(source.read_text(encoding="utf-8"))

    assert generated["ingest"] == original["ingest"]
    assert generated["project"] == original["project"]
    assert generated["paths"]["data_root"] == "data"
    assert generated["paths"]["models_dir"] == "/models"
    parsed = load_config(destination)
    assert parsed.ingest == load_config(source).ingest


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
