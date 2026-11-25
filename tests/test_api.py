"""Exercise real local HTTP requests without downloading model weights."""

import json
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from src.api import RagHTTPServer
from src.api_service import ApiConfig, QueryRequest, RagService
from src.indexer import save_index
from src.models import Chunk, ChunkIndex, ScoredChunk


@pytest.fixture
def service() -> RagService:
    """Keep two retrieved passages and a predictable token budget."""
    retriever = Mock()
    retriever.search.return_value = [
        ScoredChunk(
            chunk=Chunk(
                file_path=f"corpus/{number}.txt",
                first_character_index=0,
                last_character_index=7,
                text="adapter",
            ),
            score=1.0 / number,
        )
        for number in (1, 2)
    ]
    return RagService(ApiConfig(context_budget=400), retriever=retriever)


@pytest.fixture
def generator() -> Mock:
    """Simulate ten instruction tokens plus one hundred per source."""
    result = Mock()
    result.context_budget = 400
    result.max_new_tokens = 256
    result.generate.return_value = "Use the adapter [Source 1]."

    def tokenize(
        messages: list[dict[str, str]], **kwargs: object,
    ) -> list[int]:
        """Allow only the first of two retrieved passages to fit."""
        sources = messages[1]["content"].count("[Source ")
        return [0] * (10 + 100 * sources)

    result.tokenizer.apply_chat_template.side_effect = tokenize
    return result


@pytest.fixture
def server(service: RagService) -> Iterator[RagHTTPServer]:
    """Serve on an ephemeral loopback port and cleanly stop after each test."""
    with RagHTTPServer(("127.0.0.1", 0), service) as httpd:
        thread = Thread(
            target=httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        try:
            yield httpd
        finally:
            httpd.shutdown()
            thread.join(timeout=5)


def request(
    server: RagHTTPServer,
    method: str,
    route: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Send an actual HTTP request and decode its JSON response."""
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    try:
        connection.request(method, route, body, headers or {})
        response = connection.getresponse()
        assert response.getheader("Content-Type") == (
            "application/json; charset=utf-8"
        )
        payload = json.loads(response.read())
        return response.status, cast(dict[str, Any], payload)
    finally:
        connection.close()


def post(
    server: RagHTTPServer, route: str, payload: object,
) -> tuple[int, dict[str, Any]]:
    """Send a JSON query to a POST endpoint."""
    return request(
        server, "POST", route, json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )


def test_health_does_not_load_generator(server: RagHTTPServer) -> None:
    """Readiness confirms the index is loaded, without loading Qwen."""
    status, payload = request(server, "GET", "/health")

    assert status == 200
    assert payload == {
        "status": "ok", "method": "bm25", "device": "cpu",
        "generator_loaded": False,
    }


def test_search_returns_scores_and_source_ranges(
    server: RagHTTPServer,
) -> None:
    """Preserve score, passage and provenance in the HTTP response."""
    status, payload = post(server, "/search", {"question": "adapter", "k": 2})

    assert status == 200
    assert payload["method"] == "bm25"
    assert payload["results"][0]["score"] == 1.0
    assert payload["results"][0]["chunk"] == {
        "file_path": "corpus/1.txt", "first_character_index": 0,
        "last_character_index": 7, "text": "adapter",
    }


@pytest.mark.parametrize("payload", [
    {}, {"question": " "}, {"question": 3},
    {"question": "adapter", "k": -1},
    {"question": "adapter", "k": 21},
    {"question": "adapter", "k": True},
    {"question": "adapter", "k": "2"},
    {"question": "adapter", "model_name": "other"},
    {"question": "x" * 16001}, [],
])
def test_invalid_query_returns_422(
    server: RagHTTPServer, payload: object,
) -> None:
    """Reject bad fields before invoking the retriever."""
    status, body = post(server, "/search", payload)

    assert status == 422
    assert "error" in body
    cast(Mock, server.service.retriever).search.assert_not_called()


@pytest.mark.parametrize(
    ("method", "route", "expected"),
    [("GET", "/search", 405), ("POST", "/health", 405),
     ("GET", "/missing", 404), ("POST", "/missing", 404),
     ("PUT", "/search", 501)],
)
def test_wrong_route_or_method(
    server: RagHTTPServer, method: str, route: str, expected: int,
) -> None:
    """Report route and method errors as JSON."""
    status, payload = request(server, method, route)

    assert status == expected
    assert "error" in payload


@pytest.mark.parametrize(
    ("body", "headers", "expected"),
    [
        (b"{broken", {"Content-Type": "application/json"}, 400),
        (b"\xff", {"Content-Type": "application/json"}, 400),
        (b"{}", {"Content-Type": "text/plain"}, 415),
        (b"{}", {"Content-Length": "-1"}, 400),
        (b"{}", {"Content-Length": "invalid"}, 400),
        (b"x" * 65537, {"Content-Type": "application/json"}, 413),
        (b"{}", {"Transfer-Encoding": "chunked"}, 400),
    ],
)
def test_invalid_request_body(
    server: RagHTTPServer, body: bytes,
    headers: dict[str, str], expected: int,
) -> None:
    """Reject malformed or oversized request bodies."""
    status, payload = request(server, "POST", "/search", body, headers)

    assert status == expected
    assert "error" in payload


def test_answer_loads_cpu_generator_once_and_limits_sources(
    server: RagHTTPServer, generator: Mock,
) -> None:
    """Reuse the model and return only evidence fitting the input budget."""
    with patch("src.generator.Generator", return_value=generator) as factory:
        for _ in range(2):
            status, payload = post(server, "/answer", {"question": "adapter"})
            assert status == 200
            assert payload["answer"] == "Use the adapter [Source 1]."
            assert [source["file_path"] for source in payload["sources"]] == [
                "corpus/1.txt",
            ]

    factory.assert_called_once_with(
        model_name="Qwen/Qwen3-0.6B", context_budget=400,
        max_new_tokens=256, device="cpu",
    )
    assert generator.generate.call_count == 2


def test_no_hits_does_not_load_model(server: RagHTTPServer) -> None:
    """An empty retrieval returns insufficient evidence without Qwen."""
    cast(Mock, server.service.retriever).search.return_value = []
    with patch("src.generator.Generator") as factory:
        status, payload = post(server, "/answer", {"question": "unknown"})

    assert status == 200
    assert payload["sources"] == []
    assert "Insufficient evidence" in payload["answer"]
    factory.assert_not_called()


def test_no_fitting_sources_skips_generation(
    server: RagHTTPServer, generator: Mock,
) -> None:
    """Do not generate if no complete retrieved passage fits."""
    generator.context_budget = 300
    server.service.generator = generator

    status, payload = post(server, "/answer", {"question": "adapter"})

    assert status == 200
    assert payload["sources"] == []
    generator.generate.assert_not_called()


def test_model_load_failure_is_retryable(server: RagHTTPServer) -> None:
    """Keep the server usable if the first model load fails."""
    with patch("src.generator.Generator", side_effect=OSError("missing")):
        status, payload = post(server, "/answer", {"question": "adapter"})

    assert status == 503
    assert "error" in payload
    assert post(server, "/search", {"question": "adapter"})[0] == 200


def test_generation_failure_releases_work_lock(
    server: RagHTTPServer, generator: Mock,
) -> None:
    """Hide internal failure details and release the request lock."""
    generator.generate.side_effect = RuntimeError("internal details")
    server.service.generator = generator

    status, payload = post(server, "/answer", {"question": "adapter"})

    assert status == 500
    assert payload == {"error": "RAG processing failed"}
    assert post(server, "/search", {"question": "adapter"})[0] == 200


def test_busy_server_keeps_health_available(server: RagHTTPServer) -> None:
    """Reject overlapping model work while keeping readiness responsive."""
    with server.service.work_lock:
        assert post(server, "/answer", {"question": "adapter"})[0] == 503
        assert request(server, "GET", "/health")[0] == 200


def test_question_above_token_budget_returns_422(
    server: RagHTTPServer, generator: Mock,
) -> None:
    """Reject a prompt that cannot fit even without retrieved passages."""
    generator.tokenizer.apply_chat_template.side_effect = None
    generator.tokenizer.apply_chat_template.return_value = [0] * 200
    server.service.generator = generator

    assert post(server, "/answer", {"question": "adapter"})[0] == 422
    generator.generate.assert_not_called()


def test_real_bm25_service_loads_index_once(tmp_path: Path) -> None:
    """Reuse a real lexical index across searches without reading it again."""
    texts = ["adapter setup", "server port", "cache memory"]
    index = ChunkIndex(max_chunk_size=2000, chunks=[
        Chunk(
            file_path=f"{number}.txt", first_character_index=0,
            last_character_index=len(text), text=text,
        )
        for number, text in enumerate(texts)
    ])
    target = save_index(index, tmp_path)
    service = RagService(ApiConfig(processed_dir=tmp_path))
    target.unlink()

    for _ in range(2):
        response = service.execute("/search", QueryRequest(question="adapter"))
        assert response.model_dump()["results"][0]["chunk"]["text"] == (
            "adapter setup"
        )


def test_missing_index_fails_at_startup(tmp_path: Path) -> None:
    """Do not start a server claiming readiness without an index."""
    with pytest.raises(FileNotFoundError):
        RagService(ApiConfig(processed_dir=tmp_path))


def test_invalid_generation_budget_fails_at_startup() -> None:
    """Reject incompatible settings before accepting HTTP requests."""
    with pytest.raises(ValueError, match="below context_budget"):
        RagService(ApiConfig(context_budget=100), retriever=Mock())
