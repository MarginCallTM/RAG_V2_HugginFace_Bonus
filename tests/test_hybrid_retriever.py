"""Check hybrid retrieval orchestration without loading any models."""

from typing import cast
from unittest.mock import Mock

import pytest

from src.hybrid_retriever import HybridRetriever
from src.models import Chunk, ScoredChunk


def make_hit(name: str) -> ScoredChunk:
    """Create a controlled source for checking fused output provenance."""
    return ScoredChunk(
        chunk=Chunk(
            file_path=f"data/raw/{name}.txt",
            first_character_index=100,
            last_character_index=100 + len(name),
            text=name,
        ),
        score=0.5,
    )


@pytest.mark.parametrize("rank_constant", [0, 60])
def test_hybrid_requests_candidates_and_returns_fused_top_k(
    rank_constant: int,
) -> None:
    """Retrieve both candidate lists and apply the configured RRF constant."""
    a, b, c, d = [make_hit(name) for name in "ABCD"]
    lexical, semantic = Mock(), Mock()
    lexical.search.return_value = [a, b, c]
    semantic.search.return_value = [b, d, a]
    retriever = HybridRetriever(
        lexical, semantic, candidate_k=20, rank_constant=rank_constant,
    )

    results = retriever.search("example question", k=2)

    lexical.search.assert_called_once_with("example question", k=20)
    semantic.search.assert_called_once_with("example question", k=20)
    assert [hit.chunk for hit in results] == [b.chunk, a.chunk]
    assert [hit.score for hit in results] == pytest.approx([
        1 / (rank_constant + 2) + 1 / (rank_constant + 1),
        1 / (rank_constant + 1) + 1 / (rank_constant + 3),
    ])


def test_hybrid_expands_candidates_for_a_larger_final_k() -> None:
    """Request at least k candidates per method without padding results."""
    lexical, semantic = Mock(), Mock()
    a = make_hit("A")
    lexical.search.return_value = [a]
    semantic.search.return_value = []
    retriever = HybridRetriever(lexical, semantic, candidate_k=20)

    results = retriever.search("question", k=50)

    lexical.search.assert_called_once_with("question", k=50)
    semantic.search.assert_called_once_with("question", k=50)
    assert [hit.chunk for hit in results] == [a.chunk]


@pytest.mark.parametrize("empty_method", ["lexical", "semantic", "both"])
def test_hybrid_handles_missing_candidates(empty_method: str) -> None:
    """Allow either method or both methods to return no candidates."""
    a = make_hit("A")
    lexical, semantic = Mock(), Mock()
    lexical.search.return_value = (
        [] if empty_method in ("lexical", "both") else [a]
    )
    semantic.search.return_value = (
        [] if empty_method in ("semantic", "both") else [a]
    )

    results = HybridRetriever(lexical, semantic).search("question")

    assert [hit.chunk for hit in results] == (
        [] if empty_method == "both" else [a.chunk]
    )


@pytest.mark.parametrize(
    ("query", "k"), [("question", 0), ("", 5), (" \n\t", 5)],
)
def test_hybrid_skips_unnecessary_searches(query: str, k: int) -> None:
    """Avoid both backends for zero results or a blank question."""
    lexical, semantic = Mock(), Mock()

    assert HybridRetriever(lexical, semantic).search(query, k=k) == []
    lexical.search.assert_not_called()
    semantic.search.assert_not_called()


@pytest.mark.parametrize("invalid", [-1, 1.5, "5", True])
def test_hybrid_rejects_invalid_search_k(invalid: object) -> None:
    """Reject malformed runtime k values before either backend is called."""
    lexical, semantic = Mock(), Mock()

    with pytest.raises(ValueError, match="k must be"):
        HybridRetriever(lexical, semantic).search(
            "question", k=cast(int, invalid),
        )

    lexical.search.assert_not_called()
    semantic.search.assert_not_called()


@pytest.mark.parametrize("invalid", [0, -1, 1.5, "20", True])
def test_hybrid_rejects_invalid_candidate_count(invalid: object) -> None:
    """Require a positive integer candidate count."""
    with pytest.raises(ValueError, match="candidate_k must be"):
        HybridRetriever(Mock(), Mock(), candidate_k=cast(int, invalid))


@pytest.mark.parametrize("invalid", [-1, 1.5, "60", True])
def test_hybrid_rejects_invalid_rank_constant(invalid: object) -> None:
    """Reject invalid smoothing constants before any retrieval."""
    with pytest.raises(ValueError, match="rank_constant must be"):
        HybridRetriever(Mock(), Mock(), rank_constant=cast(int, invalid))


@pytest.mark.parametrize("failing_method", ["lexical", "semantic"])
def test_hybrid_propagates_backend_failures(failing_method: str) -> None:
    """Do not silently return a partial ranking when a backend fails."""
    lexical, semantic = Mock(), Mock()
    lexical.search.return_value = []
    semantic.search.return_value = []
    failing = lexical if failing_method == "lexical" else semantic
    failing.search.side_effect = ValueError("backend failure")

    with pytest.raises(ValueError, match="backend failure"):
        HybridRetriever(lexical, semantic).search("question")
