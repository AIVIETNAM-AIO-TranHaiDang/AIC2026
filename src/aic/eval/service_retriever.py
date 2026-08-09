"""Evaluate the deployed retrieval service through its public search API."""

from __future__ import annotations

from typing import Protocol

from aic.eval.fixture import QueryCase
from aic.retrieval.base import SearchResult


class _Response(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> dict: ...


class _Client(Protocol):
    def post(self, path: str, *, json: dict) -> _Response: ...

    def close(self) -> None: ...


class ServiceRetriever:
    """Retriever adapter for an already-running ``/api/search`` service.

    This keeps the normal fixture schema and Recall@K implementation while
    evaluating the exact deployment an operator uses. It deliberately does
    not support KIS-V because the current service endpoint accepts text, not
    an example video upload.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        client: _Client | None = None,
    ) -> None:
        if client is None:
            import httpx

            client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)
        self._client = client

    def close(self) -> None:
        self._client.close()

    def search(self, case: QueryCase, top_k: int) -> list[SearchResult]:
        if case.kind == "kis_v":
            raise ValueError("remote service evaluation does not support KIS-V")
        text = "\n".join(
            part for part in [case.text or "", *case.reveals] if part
        ).strip()
        if not text:
            raise ValueError(f"case {case.query_id}: no text query for /api/search")
        response = self._client.post(
            "/api/search",
            json={"query": text, "top_k": top_k},
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("results")
        if not isinstance(rows, list):
            raise ValueError("/api/search response has no results list")
        return [
            SearchResult(
                video_id=str(row["video_id"]),
                timestamp_ms=int(row["timestamp_ms"]),
                score=float(row["score"]),
                keyframe_id=(
                    str(row["keyframe_id"])
                    if row.get("keyframe_id") is not None
                    else None
                ),
            )
            for row in rows[:top_k]
        ]
