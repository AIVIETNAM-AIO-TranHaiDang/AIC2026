"""Per-stage JSONL manifests: the resumability backbone.

Every batch stage writes one record per produced artifact to an append-only
JSONL manifest and reads it back on restart to skip completed work. Records are
flushed line by line so a killed process (a dying Colab session, a Ctrl-C)
loses at most the record being written; a trailing partial line is tolerated
and ignored on load.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Any

logger = logging.getLogger(__name__)


class ManifestError(ValueError):
    """Raised when a manifest record or file is invalid."""


def read_manifest(path: Path | str) -> Iterator[dict[str, Any]]:
    """Yield records from a JSONL manifest, tolerating a torn final line."""
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning(
                    "manifest %s: ignoring unparseable line %d (torn write?)",
                    path,
                    line_no,
                )
                continue
            if not isinstance(record, dict):
                raise ManifestError(
                    f"manifest {path} line {line_no}: expected an object, "
                    f"got {type(record).__name__}"
                )
            yield record


def shard_manifest_path(path: Path | str, tag: str) -> Path:
    """Per-shard sibling of a manifest: ``foo.jsonl`` -> ``foo.<tag>.jsonl``.

    An empty ``tag`` returns the path unchanged, so the single-process path
    writes the canonical manifest exactly as before. Data-parallel workers pass
    a distinct tag (``rank0`` ...) so their appends never interleave into one
    file; a merge step concatenates the shards back afterwards.
    """
    path = Path(path)
    if not tag:
        return path
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def completed_keys(path: Path | str, key_field: str) -> set[str]:
    """Return the set of key values already present in a manifest."""
    keys: set[str] = set()
    for record in read_manifest(path):
        value = record.get(key_field)
        if isinstance(value, str):
            keys.add(value)
    return keys


class ManifestWriter:
    """Append-only JSONL writer with per-line flush.

    Usable as a context manager. ``key_field`` names the record field that
    identifies an artifact; appending a record without it is an error, which
    keeps manifests joinable and resumable by construction.
    """

    def __init__(self, path: Path | str, key_field: str) -> None:
        self._path = Path(path)
        self._key_field = key_field
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: dict[str, Any]) -> None:
        if self._key_field not in record:
            raise ManifestError(
                f"record is missing key field {self._key_field!r}: {record!r}"
            )
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> ManifestWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def rewrite_manifest(path: Path | str, keep: Callable[[dict], bool]) -> int:
    """Rewrite a JSONL manifest keeping only rows where ``keep`` is true.

    Returns the number of rows dropped. Writes atomically (temp file, then
    replace) and leaves the file untouched when no row is removed. This is
    the one sanctioned way to remove rows from an append-only manifest (the
    verify/repair pass and the explicit redo flags); ordinary stages only
    ever append.
    """
    path = Path(path)
    if not path.is_file():
        return 0
    rows = list(read_manifest(path))
    survivors = [row for row in rows if keep(row)]
    dropped = len(rows) - len(survivors)
    if dropped == 0:
        return 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in survivors:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    return dropped


def write_json_atomic(path: Path | str, payload: Any) -> None:
    """Write JSON to a file atomically (write temp, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp_path, path)
