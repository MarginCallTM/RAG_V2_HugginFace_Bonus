"""Tests for lexical retrieval on a controlled corpus."""

import pytest

from src.models import Chunk, ChunkIndex
from src.retriever import Retriever


@pytest.fixture
def retriever() -> Retriever:
    """Build a small corpus with identifiable source paths."""
    texts = [
        "load adapter",
        "install python",
        "configure server",
        "manage memory",
    ]

    chunks = [
        Chunk(
            file_path=f"data/raw/example_{number}.txt",
            first_character_index=0,
            last_character_index=len(text),
            text=text,
        )
        for number, text in enumerate(texts)
    ]

    index = ChunkIndex(max_chunk_size=2000, chunks=chunks)

    return Retriever(index)


def test_search_returns_matching_chunk_and_source(
    retriever: Retriever,
) -> None:
    """Associate the matching score with the correct source chunk."""
    results = retriever.search("adapter", k=5)

    assert len(results) == 1
    assert results[0].chunk.text == "load adapter"
    assert results[0].chunk.file_path == "data/raw/example_0.txt"
    assert results[0].score > 0


def test_search_respects_k(retriever: Retriever) -> None:
    """Return only the requested number of positive-score results."""
    results = retriever.search("adapter python server memory", k=2)

    assert len(results) == 2
    assert all(result.score > 0 for result in results)


def test_search_sorts_scores_descending(retriever: Retriever) -> None:
    """Rank a chunk matching both query terms ahead of a single match."""
    results = retriever.search("load adapter python", k=4)

    assert len(results) == 2
    assert results[0].chunk.text == "load adapter"
    assert results[0].score > results[1].score


@pytest.mark.parametrize("query", ["", "   ", "?!", "xqzv"])
def test_search_handles_queries_without_matches(
    retriever: Retriever,
    query: str,
) -> None:
    """Avoid returning arbitrary chunks for empty or unknown terms."""
    assert retriever.search(query, k=5) == []


@pytest.mark.parametrize("k", [0, -1])
def test_search_handles_non_positive_k(
    retriever: Retriever,
    k: int,
) -> None:
    """Follow the retriever's empty-result policy for non-positive k."""
    assert retriever.search("adapter", k=k) == []


def test_search_preserves_order_for_equal_scores(
    retriever: Retriever,
) -> None:
    """Keep corpus order when matching chunks receive equal scores."""
    results = retriever.search("adapter python server memory", k=4)

    assert [result.chunk.file_path for result in results] == [
        "data/raw/example_0.txt",
        "data/raw/example_1.txt",
        "data/raw/example_2.txt",
        "data/raw/example_3.txt",
    ]
    assert len({result.score for result in results}) == 1


def test_retriever_rejects_empty_index() -> None:
    """Reject an index containing no chunks."""
    index = ChunkIndex(max_chunk_size=2000, chunks=[])

    with pytest.raises(ValueError, match="empty index"):
        Retriever(index)


def test_retriever_rejects_index_without_searchable_terms() -> None:
    """Reject text that produces no lexical terms."""
    chunk = Chunk(
        file_path="data/raw/punctuation.txt",
        first_character_index=0,
        last_character_index=3,
        text="?!.",
    )
    index = ChunkIndex(max_chunk_size=2000, chunks=[chunk])

    with pytest.raises(ValueError, match="no searchable terms"):
        Retriever(index)
