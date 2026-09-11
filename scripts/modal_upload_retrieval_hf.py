"""Export the retrieval-ready AIC2026 artifacts from Modal to Hugging Face.

This intentionally does not upload source videos, JPG frames, raw embedding
shards, model caches, or the already-published ``packages/`` keyframe TARs.

Audit the source without modifying Hugging Face:
    modal run scripts/modal_upload_retrieval_hf.py::audit \
      --repo-id Neezidow/AIC2026-keyframes

Start a resumable upload (it continues safely after interruption):
    modal run --detach scripts/modal_upload_retrieval_hf.py::run \
      --repo-id Neezidow/AIC2026-keyframes
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import modal

HF_SECRET_NAME = "huggingface-aic"
HF_HUB_VERSION = "1.27.0"
VOLUME_ROOT = Path("/artifacts")
REPO_PREFIX = "retrieval/v1"
UPLOAD_BATCH_FILES = 20

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    f"huggingface_hub=={HF_HUB_VERSION}"
)
app = modal.App("aic-retrieval-huggingface")
artifacts = modal.Volume.from_name("aic-embeddings").with_mount_options(
    read_only=True
)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


@dataclass(frozen=True)
class UploadEntry:
    source_path: Path
    path_in_repo: str
    size: int
    sha256: str


def _token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token or not token.startswith("hf_"):
        raise RuntimeError("Modal secret must provide a valid HF_TOKEN")
    return token


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_specs() -> tuple[tuple[str, str], ...]:
    """(path below aic-embeddings, desired path below retrieval/v1)."""
    return (
        ("manifests/videos.jsonl", "manifests/videos.jsonl"),
        ("manifests/shots.jsonl", "manifests/shots.jsonl"),
        ("manifests/keyframes.jsonl", "manifests/keyframes.jsonl"),
        ("manifests/ocr.jsonl", "manifests/ocr.jsonl"),
        ("manifests/asr.jsonl", "manifests/asr.jsonl"),
        ("manifests/chronicle.jsonl", "manifests/chronicle.jsonl"),
        (
            "indexes/keyframes-google--siglip2-so400m-patch16-384/vectors.npy",
            "visual/vectors.npy",
        ),
        (
            "indexes/keyframes-google--siglip2-so400m-patch16-384/ids.json",
            "visual/ids.json",
        ),
        (
            "indexes/keyframes-google--siglip2-so400m-patch16-384/meta.json",
            "visual/meta.json",
        ),
        (
            "embeddings/keyframes/google--siglip2-so400m-patch16-384/manifest.jsonl",
            "visual/embedding-manifest.jsonl",
        ),
        ("indexes/chronicle-semantic/vectors.npy", "text/chronicle-semantic/vectors.npy"),
        ("indexes/chronicle-semantic/ids.json", "text/chronicle-semantic/ids.json"),
        ("indexes/chronicle-semantic/meta.json", "text/chronicle-semantic/meta.json"),
        ("indexes/chronicle-literal/docs.json", "text/chronicle-literal/docs.json"),
        ("indexes/chronicle-overlay/docs.json", "text/chronicle-overlay/docs.json"),
    )


def _collect_entries() -> list[UploadEntry]:
    entries: list[UploadEntry] = []
    missing: list[str] = []
    for source_relative, destination_relative in _source_specs():
        source_path = VOLUME_ROOT / source_relative
        if not source_path.is_file():
            missing.append(source_relative)
            continue
        entries.append(
            UploadEntry(
                source_path=source_path,
                path_in_repo=f"{REPO_PREFIX}/{destination_relative}",
                size=source_path.stat().st_size,
                sha256=_sha256(source_path),
            )
        )
    if missing:
        raise RuntimeError(f"required Modal artifacts are missing: {missing}")
    return entries


def _readme(entries: list[UploadEntry]) -> bytes:
    total_bytes = sum(entry.size for entry in entries)
    lines = [
        "# AIC2026 retrieval artifacts — v1",
        "",
        "This folder is an export of the tested retrieval state from Modal.",
        "It deliberately excludes raw videos, raw embedding shards, model cache,",
        "and JPG keyframes. The latter are already published in `packages/`.",
        "",
        f"Files: {len(entries)}; total size: {total_bytes:,} bytes.",
        "",
        "- `manifests/`: video, shot, keyframe, OCR, ASR, and combined Chronicle metadata.",
        "- `visual/`: SigLIP2 visual vectors and their row/id/temporal metadata.",
        "- `text/`: BGE-M3 semantic vectors plus literal and overlay documents.",
        "",
        "`index.faiss` is retained in Modal as a rebuild/compatibility artifact but is",
        "not included here: the current UI constructs the CPU FAISS Flat index from",
        "`vectors.npy` at start-up.",
        "",
        "Source model: `google/siglip2-so400m-patch16-384` for visual retrieval.",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _remote_files(api: object, repo_id: str) -> dict[str, object]:
    return {
        item.path: item
        for item in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            recursive=True,
            expand=True,
        )
        if hasattr(item, "size")
    }


def _entry_matches_remote(entry: UploadEntry, remote: object | None) -> bool:
    if remote is None or getattr(remote, "size", None) != entry.size:
        return False
    lfs = getattr(remote, "lfs", None)
    remote_sha256 = getattr(lfs, "sha256", None)
    return not isinstance(remote_sha256, str) or remote_sha256 == entry.sha256


@app.function(
    image=image,
    cpu=2.0,
    memory=4_096,
    secrets=[hf_secret],
    timeout=60 * 60,
    volumes={"/artifacts": artifacts},
)
def inspect_source() -> dict[str, object]:
    """Verify every intended artifact exists and record exact byte totals."""
    entries = _collect_entries()
    return {
        "status": "ready",
        "files": len(entries),
        "bytes": sum(entry.size for entry in entries),
        "entries": [
            {"path": entry.path_in_repo, "bytes": entry.size}
            for entry in entries
        ],
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=8_192,
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    retries=2,
    max_containers=1,
    volumes={"/artifacts": artifacts},
    env={"HF_XET_HIGH_PERFORMANCE": "1", "HF_XET_CACHE": "/tmp/hf-xet-cache"},
)
def upload(repo_id: str) -> dict[str, object]:
    """Upload missing/different artifacts in bounded commits and verify them."""
    from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415

    api = HfApi(token=_token())
    api.auth_check(repo_id=repo_id, repo_type="dataset", write=True)
    entries = _collect_entries()
    remote = _remote_files(api, repo_id)
    pending = [
        entry for entry in entries if not _entry_matches_remote(entry, remote.get(entry.path_in_repo))
    ]
    print(
        f"verified source: {len(entries)} files; pending upload: {len(pending)} files",
        flush=True,
    )

    for offset in range(0, len(pending), UPLOAD_BATCH_FILES):
        group = pending[offset : offset + UPLOAD_BATCH_FILES]
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=(
                f"Upload AIC2026 retrieval artifacts v1 "
                f"({offset // UPLOAD_BATCH_FILES + 1}/"
                f"{(len(pending) + UPLOAD_BATCH_FILES - 1) // UPLOAD_BATCH_FILES})"
            ),
            operations=[
                CommitOperationAdd(
                    path_in_repo=entry.path_in_repo,
                    path_or_fileobj=entry.source_path,
                )
                for entry in group
            ],
            num_threads=8,
        )
        print(f"committed {offset + len(group)}/{len(pending)} pending files", flush=True)

    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Document AIC2026 retrieval artifacts v1",
        operations=[
            CommitOperationAdd(
                path_in_repo=f"{REPO_PREFIX}/README.md",
                path_or_fileobj=_readme(entries),
            )
        ],
        num_threads=1,
    )

    final_remote = _remote_files(api, repo_id)
    mismatched = [
        entry.path_in_repo
        for entry in entries
        if not _entry_matches_remote(entry, final_remote.get(entry.path_in_repo))
    ]
    if mismatched:
        raise RuntimeError(f"Hugging Face verification failed: {mismatched}")
    result = {
        "status": "done",
        "repo_id": repo_id,
        "files": len(entries),
        "bytes": sum(entry.size for entry in entries),
        "uploaded_this_run": len(pending),
        "prefix": REPO_PREFIX,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint(name="audit")
def audit() -> None:
    print(json.dumps(inspect_source.remote(), ensure_ascii=False, indent=2))


@app.local_entrypoint(name="run")
def run(repo_id: str = "Neezidow/AIC2026-keyframes") -> None:
    call = upload.spawn(repo_id)
    print(
        f"submitted Modal-to-Hugging-Face retrieval upload for {repo_id}; "
        f"function_call_id={call.object_id}"
    )
