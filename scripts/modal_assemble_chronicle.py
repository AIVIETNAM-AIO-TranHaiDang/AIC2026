"""Assemble the persisted OCR and ASR manifests into Chronicle on Modal.

This job is intentionally CPU-only: extraction has already completed, so it
calls the source ``assemble_chronicle`` function directly and validates the
written records before committing the Volume.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(LOCAL_ROOT / "requirements.txt"))
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
    .add_local_dir(
        LOCAL_ROOT / "configs", remote_path=str(REMOTE_ROOT / "configs"), copy=True
    )
)

app = modal.App("aic-assemble-chronicle")
results = modal.Volume.from_name("aic-embeddings", create_if_missing=False)


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=60 * 60,
    volumes={"/mnt/out": results},
)
def assemble() -> dict[str, int]:
    """Join shots, OCR, and ASR; then validate every Chronicle row."""
    os.chdir(REMOTE_ROOT)
    sys.path.insert(0, str(REMOTE_ROOT / "src"))

    data_dir = REMOTE_ROOT / "data"
    manifests_link = data_dir / "manifests"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifests_link.symlink_to("/mnt/out/manifests", target_is_directory=True)

    from aic.chronicle.jobs import assemble_chronicle, load_chronicle
    from aic.config import load_config

    cfg = load_config(REMOTE_ROOT / "configs" / "t0.yaml")
    stats = assemble_chronicle(cfg)
    records = load_chronicle(cfg)
    with_asr = sum(record.asr_text is not None for record in records)
    with_ocr = sum(bool(record.ocr) for record in records)

    if len(records) != stats.processed:
        raise RuntimeError(
            "validation count mismatch: "
            f"assembled={stats.processed}, loaded={len(records)}"
        )

    results.commit()
    return {
        "shots": len(records),
        "shots_with_asr": with_asr,
        "shots_with_ocr": with_ocr,
    }


@app.local_entrypoint()
def main() -> None:
    print(assemble.remote())
