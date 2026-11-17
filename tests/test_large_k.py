"""Check oversized retrieval requests on a small controlled index."""

import pytest

from src.models import Chunk, ChunkIndex
from src.retriever import Retriever


@pytest.mark.parametrize("k", [5, 100, 10**6, 10**100])
@pytest.mark.parametrize(
    ("query", "expected_positions"),
    [
        ("adapter python server memory", [0, 1, 2, 3]),
        ("adapter", [0]),
        ("unknownterm", []),
    ],
)
def test_large_k_returns_only_available_matches(
    k: int,
    query: str,
    expected_positions: list[int],
) -> None:
    """Preserve ranking and exclude nonmatches without padding results."""
    texts = [
        "load adapter",
        "install python",
        "configure server",
        "manage memory",
    ]
    chunks = [
        Chunk(
            file_path=f"data/raw/example_{position}.txt",
            first_character_index=0,
            last_character_index=len(text),
            text=text,
        )
        for position, text in enumerate(texts)
    ]
    retriever = Retriever(ChunkIndex(max_chunk_size=2000, chunks=chunks))

    results = retriever.search(query, k=k)

    expected = [chunks[position] for position in expected_positions]
    assert [result.chunk for result in results] == expected
    assert len(results) <= len(chunks)
    assert all(result.score > 0 for result in results)
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )
