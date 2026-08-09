from __future__ import annotations

import pytest

from aic.eval.fixture import GroundTruth, QueryCase
from aic.eval.service_retriever import ServiceRetriever


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "results": [
                {
                    "video_id": "L21_V001",
                    "timestamp_ms": 1234,
                    "score": 0.75,
                },
                {
                    "video_id": "L21_V002",
                    "timestamp_ms": 5678,
                    "score": 0.5,
                },
            ]
        }


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.closed = False

    def post(self, path: str, *, json: dict) -> FakeResponse:
        self.calls.append((path, json))
        return FakeResponse()

    def close(self) -> None:
        self.closed = True


def case(kind: str = "kis_t") -> QueryCase:
    fields = {
        "query_id": "q1",
        "kind": kind,
        "truths": [GroundTruth(video_id="L21_V001", t_start_ms=0, t_end_ms=2000)],
    }
    if kind == "kis_v":
        fields["clip_path"] = "query.mp4"
    elif kind == "kis_c":
        fields["text"] = "người mặc áo xanh"
        fields["reveals"] = ["đứng cạnh một chiếc xe đỏ"]
    else:
        fields["text"] = "người mặc áo xanh"
    return QueryCase.model_validate(fields)


def test_service_retriever_uses_fixture_query_and_top_k() -> None:
    client = FakeClient()
    retriever = ServiceRetriever("https://example.test", client=client)

    results = retriever.search(case("kis_c"), top_k=1)

    assert client.calls == [
        (
            "/api/search",
            {
                "query": "người mặc áo xanh\nđứng cạnh một chiếc xe đỏ",
                "top_k": 1,
            },
        )
    ]
    assert len(results) == 1
    assert results[0].video_id == "L21_V001"
    assert results[0].timestamp_ms == 1234
    assert results[0].score == 0.75


def test_service_retriever_rejects_kis_v() -> None:
    retriever = ServiceRetriever("https://example.test", client=FakeClient())

    with pytest.raises(ValueError, match="does not support KIS-V"):
        retriever.search(case("kis_v"), top_k=10)
