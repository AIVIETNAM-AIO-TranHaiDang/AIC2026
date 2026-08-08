"""Upload verified keyframe packages from a Modal Volume to Hugging Face.

Check token/repository write access without changing the repo:
    modal run scripts/modal_upload_hf.py::check \
      --repo-id Neezidow/AIC2026-keyframes

Upload one real report as a small end-to-end pilot:
    modal run scripts/modal_upload_hf.py::pilot \
      --repo-id Neezidow/AIC2026-keyframes

Submit the full resumable CPU upload:
    modal run --detach scripts/modal_upload_hf.py::run \
      --repo-id Neezidow/AIC2026-keyframes
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

import modal

LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/aic")
RESULTS_ROOT = Path("/results")
HF_SECRET_NAME = "huggingface-aic"
HF_HUB_VERSION = "1.27.0"
UPLOAD_BATCH_FILES = 50

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"huggingface_hub=={HF_HUB_VERSION}")
    .add_local_dir(LOCAL_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"), copy=True)
)
app = modal.App("aic-keyframes-huggingface")
results = modal.Volume.from_name("aic-keyframe-results").with_mount_options(
    read_only=True
)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)


def _imports() -> tuple[object, object, object, object, object, object]:
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415

    from aic.hf_keyframes import (  # noqa: PLC0415
        batches,
        build_dataset_index,
        build_readme,
        collect_upload_entries,
    )

    return (
        HfApi,
        CommitOperationAdd,
        batches,
        build_dataset_index,
        build_readme,
        collect_upload_entries,
    )


def _token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token or not token.startswith("hf_"):
        raise RuntimeError("Modal secret must provide a valid HF_TOKEN")
    return token


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


def _remote_sha256(item: object) -> str | None:
    lfs = getattr(item, "lfs", None)
    checksum = getattr(lfs, "sha256", None)
    return checksum if isinstance(checksum, str) else None


def _entry_matches_remote(entry: object, remote: object | None) -> bool:
    if remote is None or getattr(remote, "size", None) != entry.size:
        return False
    checksum = _remote_sha256(remote)
    return checksum is None or checksum == entry.sha256


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tar_jsonl(bundle: tarfile.TarFile, name: str) -> list[dict[str, object]]:
    handle = bundle.extractfile(name)
    if handle is None:
        raise RuntimeError(f"cannot read TAR member: {name}")
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(handle, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {name}:{line_number}") from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"non-object JSONL row at {name}:{line_number}")
        rows.append(row)
    return rows


@app.function(image=image, cpu=0.25, memory=512, secrets=[hf_secret], timeout=300)
def check_access(repo_id: str) -> dict[str, object]:
    """Verify identity plus repo-scoped write access without changing the repo."""
    HfApi, *_ = _imports()
    token = _token()
    api = HfApi(token=token)
    identity = api.whoami()
    api.auth_check(repo_id=repo_id, repo_type="dataset", write=True)
    info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    return {
        "status": "ok",
        "account": identity.get("name"),
        "repo_id": repo_id,
        "private": getattr(info, "private", None),
        "write_access": True,
    }


@app.function(
    image=image,
    cpu=1.0,
    memory=1_024,
    secrets=[hf_secret],
    timeout=15 * 60,
    volumes={"/results": results},
)
def upload_pilot(repo_id: str) -> dict[str, object]:
    """Upload and verify one real report before any large-file transfer."""
    HfApi, CommitOperationAdd, _, _, _, collect_upload_entries = _imports()
    token = _token()
    api = HfApi(token=token)
    api.auth_check(repo_id=repo_id, repo_type="dataset", write=True)
    entries, _ = collect_upload_entries(
        RESULTS_ROOT,
        [("Videos_L21_a", "L21_V001")],
        require_exact_files=False,
    )
    entry = next(item for item in entries if item.kind == "report")
    remote = _remote_files(api, repo_id).get(entry.path_in_repo)
    if not _entry_matches_remote(entry, remote):
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add verified AIC2026 keyframe upload pilot",
            operations=[
                CommitOperationAdd(
                    path_in_repo=entry.path_in_repo,
                    path_or_fileobj=entry.source_path,
                )
            ],
            num_threads=1,
        )
    verified = _remote_files(api, repo_id).get(entry.path_in_repo)
    if not _entry_matches_remote(entry, verified):
        raise RuntimeError(f"pilot verification failed: {entry.path_in_repo}")
    return {
        "status": "ok",
        "repo_id": repo_id,
        "path": entry.path_in_repo,
        "bytes": entry.size,
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=4_096,
    secrets=[hf_secret],
    timeout=24 * 60 * 60,
    retries=2,
    max_containers=1,
    volumes={"/results": results},
    env={
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_XET_CACHE": "/tmp/hf-xet-cache",
    },
)
def upload_all(repo_id: str) -> dict[str, object]:
    """Upload all verified packages in resumable bounded commits, then audit."""
    (
        HfApi,
        CommitOperationAdd,
        make_batches,
        build_dataset_index,
        build_readme,
        collect_upload_entries,
    ) = _imports()
    token = _token()
    api = HfApi(token=token)
    api.auth_check(repo_id=repo_id, repo_type="dataset", write=True)
    entries, reports = collect_upload_entries(RESULTS_ROOT)
    remote_files = _remote_files(api, repo_id)
    pending = [
        entry
        for entry in entries
        if not _entry_matches_remote(entry, remote_files.get(entry.path_in_repo))
    ]
    print(
        f"verified source: {len(reports)} videos, {len(entries)} files; "
        f"pending upload: {len(pending)} files",
        flush=True,
    )

    groups = make_batches(pending, UPLOAD_BATCH_FILES)
    for batch_number, group in enumerate(groups, start=1):
        operations = [
            CommitOperationAdd(
                path_in_repo=entry.path_in_repo,
                path_or_fileobj=entry.source_path,
            )
            for entry in group
        ]
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=(
                f"Upload AIC2026 keyframe packages "
                f"({batch_number}/{len(groups)})"
            ),
            operations=operations,
            num_threads=8,
        )
        completed = sum(len(batch) for batch in groups[:batch_number])
        print(
            f"committed {completed}/{len(pending)} pending files",
            flush=True,
        )

    index_bytes = build_dataset_index(entries, reports)
    generated_operations = [
        CommitOperationAdd(
            path_in_repo="dataset-index.jsonl",
            path_or_fileobj=index_bytes,
        )
    ]
    refreshed = _remote_files(api, repo_id)
    if "README.md" not in refreshed:
        generated_operations.append(
            CommitOperationAdd(
                path_in_repo="README.md",
                path_or_fileobj=build_readme(reports),
            )
        )
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add AIC2026 keyframe package catalogue",
        operations=generated_operations,
        num_threads=2,
    )

    final_files = _remote_files(api, repo_id)
    issues = []
    for entry in entries:
        remote = final_files.get(entry.path_in_repo)
        if not _entry_matches_remote(entry, remote):
            issues.append(entry.path_in_repo)
    expected_package_paths = {entry.path_in_repo for entry in entries}
    actual_package_paths = {
        path for path in final_files if path.startswith("packages/")
    }
    unexpected = sorted(actual_package_paths - expected_package_paths)
    missing = sorted(expected_package_paths - actual_package_paths)
    if issues or missing or unexpected:
        raise RuntimeError(
            "Hugging Face verification failed: "
            f"mismatched={issues[:5]}, missing={missing[:5]}, "
            f"unexpected={unexpected[:5]}"
        )
    result = {
        "status": "done",
        "repo_id": repo_id,
        "videos": len(reports),
        "package_files": len(entries),
        "tar_bytes": sum(entry.size for entry in entries if entry.kind == "tar"),
        "index_bytes": len(index_bytes),
        "uploaded_this_run": len(pending),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(
    image=image,
    cpu=2.0,
    memory=2_048,
    secrets=[hf_secret],
    timeout=60 * 60,
    volumes={"/results": results},
)
def verify_roundtrip(
    repo_id: str,
    archive: str = "Videos_L21_a",
    video_id: str = "L21_V001",
) -> dict[str, object]:
    """Download one gated package and compare it with its verified source."""
    sys.path.insert(0, str(REMOTE_ROOT / "src"))
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    from aic.hf_keyframes import collect_upload_entries  # noqa: PLC0415

    entries, source_reports = collect_upload_entries(
        RESULTS_ROOT,
        [(archive, video_id)],
        require_exact_files=False,
    )
    by_kind = {entry.kind: entry for entry in entries}
    tar_entry = by_kind["tar"]
    report_entry = by_kind["report"]
    token = _token()
    download_root = Path("/tmp/hf-roundtrip")
    downloaded_tar = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=tar_entry.path_in_repo,
            token=token,
            cache_dir=download_root / "cache",
        )
    )
    downloaded_report = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=report_entry.path_in_repo,
            token=token,
            cache_dir=download_root / "cache",
        )
    )

    if downloaded_tar.stat().st_size != tar_entry.size:
        raise RuntimeError("downloaded TAR size differs from Modal source")
    if _sha256(downloaded_tar) != tar_entry.sha256:
        raise RuntimeError("downloaded TAR SHA-256 differs from Modal source")
    if downloaded_report.stat().st_size != report_entry.size:
        raise RuntimeError("downloaded report size differs from Modal source")
    if _sha256(downloaded_report) != report_entry.sha256:
        raise RuntimeError("downloaded report SHA-256 differs from Modal source")

    downloaded_report_data = json.loads(downloaded_report.read_text(encoding="utf-8"))
    source_report = source_reports[0]
    if downloaded_report_data != source_report:
        raise RuntimeError("downloaded report content differs from Modal source")

    manifest_names = {
        "videos": "data/manifests/videos.jsonl",
        "shots": "data/manifests/shots.jsonl",
        "keyframes": "data/manifests/keyframes.jsonl",
    }
    with tarfile.open(downloaded_tar, mode="r:*") as bundle:
        members = bundle.getmembers()
        unsafe = [
            member.name
            for member in members
            if PurePosixPath(member.name).is_absolute()
            or ".." in PurePosixPath(member.name).parts
            or member.issym()
            or member.islnk()
        ]
        if unsafe:
            raise RuntimeError(f"unsafe TAR members: {unsafe[:5]}")
        file_names = {member.name for member in members if member.isfile()}
        missing_manifests = sorted(set(manifest_names.values()) - file_names)
        if missing_manifests:
            raise RuntimeError(f"missing TAR manifests: {missing_manifests}")

        image_names = sorted(
            name
            for name in file_names
            if name.startswith(f"data/keyframes/{video_id}/")
            and name.lower().endswith(".jpg")
        )
        videos_rows = _read_tar_jsonl(bundle, manifest_names["videos"])
        shots_rows = _read_tar_jsonl(bundle, manifest_names["shots"])
        keyframes_rows = _read_tar_jsonl(bundle, manifest_names["keyframes"])
        if len(videos_rows) != 1 or videos_rows[0].get("video_id") != video_id:
            raise RuntimeError(
                "videos.jsonl does not describe exactly the selected video"
            )
        if len(shots_rows) != source_report["shots"]:
            raise RuntimeError("shots.jsonl row count differs from report")
        if len(keyframes_rows) != source_report["keyframes"]:
            raise RuntimeError("keyframes.jsonl row count differs from report")
        if len(image_names) != source_report["keyframes"]:
            raise RuntimeError("JPG count differs from report")
        if any(row.get("video_id") != video_id for row in shots_rows):
            raise RuntimeError("shots.jsonl contains another video_id")
        if any(row.get("video_id") != video_id for row in keyframes_rows):
            raise RuntimeError("keyframes.jsonl contains another video_id")

        manifest_images = {str(row.get("image_path", "")) for row in keyframes_rows}
        if manifest_images != set(image_names):
            raise RuntimeError(
                "keyframes.jsonl image paths do not match TAR JPG members"
            )
        sample_positions = sorted({0, len(image_names) // 2, len(image_names) - 1})
        for position in sample_positions:
            handle = bundle.extractfile(image_names[position])
            if handle is None or handle.read(2) != b"\xff\xd8":
                raise RuntimeError(f"invalid JPEG signature: {image_names[position]}")

    result = {
        "status": "verified",
        "repo_id": repo_id,
        "archive": archive,
        "video_id": video_id,
        "tar_bytes": tar_entry.size,
        "tar_sha256": tar_entry.sha256,
        "shots": len(shots_rows),
        "keyframes": len(keyframes_rows),
        "jpg_files": len(image_names),
        "manifests": list(manifest_names.values()),
        "checks": [
            "gated_authenticated_download",
            "source_size_and_sha256_match",
            "safe_tar_paths",
            "jsonl_parse_and_counts",
            "manifest_image_paths_match",
            "jpeg_signature_samples",
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint(name="check")
def check(repo_id: str = "Neezidow/AIC2026-keyframes") -> None:
    print(json.dumps(check_access.remote(repo_id), ensure_ascii=False, indent=2))


@app.local_entrypoint(name="pilot")
def pilot(repo_id: str = "Neezidow/AIC2026-keyframes") -> None:
    print(json.dumps(upload_pilot.remote(repo_id), ensure_ascii=False, indent=2))


@app.local_entrypoint(name="run")
def run(repo_id: str = "Neezidow/AIC2026-keyframes") -> None:
    call = upload_all.spawn(repo_id)
    print(
        f"submitted Modal-to-Hugging-Face upload for {repo_id}; "
        f"function_call_id={call.object_id}"
    )


@app.local_entrypoint(name="roundtrip")
def roundtrip(
    repo_id: str = "Neezidow/AIC2026-keyframes",
    archive: str = "Videos_L21_a",
    video_id: str = "L21_V001",
) -> None:
    print(
        json.dumps(
            verify_roundtrip.remote(repo_id, archive, video_id),
            ensure_ascii=False,
            indent=2,
        )
    )
