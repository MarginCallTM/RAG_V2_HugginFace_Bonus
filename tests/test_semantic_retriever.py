"""Test semantic retrieval without loading an embedding model."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from src.embeddings import EmbeddingConfig, TokenSegment
from src.models import Chunk
from src.semantic_index import (
    SemanticIndexData,
    SemanticIndexMetaData,
    save_semantic_index,
)
from src.semantic_retriever import SemanticRetriever


@pytest.fixture
def semantic_sample(
    tmp_path: Path,
) -> tuple[Path, list[Chunk], Mock]:
    """Save controlled vectors and provide an encoder without model weights."""
    chunks = [
        Chunk(
            file_path=f"data/raw/{name}.txt",
            first_character_index=100,
            last_character_index=100 + len(name),
            text=name,
        )
        for name in ("alpha", "beta")
    ]

    segments = [
        TokenSegment(
            chunk_position=chunk_position,
            segment_position=segment_position,
            input_ids=[101, 42, 102],
        )
        for chunk_position, segment_position in [
            (0, 0),
            (0, 1),
            (1, 0),
        ]
    ]

    data = SemanticIndexData(
        metadata=SemanticIndexMetaData(
            encoder_config=EmbeddingConfig(),
            dimension=2,
            chunk_count=2,
            segment_count=3,
        ),
        chunks=chunks,
        segments=segments,
    )
    vectors = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.8, 0.6]],
        dtype=torch.float32,
    )
    target = tmp_path / "semantic_index.pt"
    save_semantic_index(data, vectors, target)

    encoder = Mock()
    encoder.dimension = 2
    encoder.encode_query.return_value = torch.tensor(
        [[1.0, 0.0]],
        dtype=torch.float32,
    )
    return target, chunks, encoder


@pytest.fixture
def semantic_retriever(
    semantic_sample: tuple[Path, list[Chunk], Mock],
) -> tuple[SemanticRetriever, list[Chunk], Mock]:
    """Load the real index while replacing only the embedding encoder."""
    target, chunks, encoder = semantic_sample

    with patch(
        "src.semantic_retriever.SemanticEncoder",
        return_value=encoder,
    ):
        retriever = SemanticRetriever(target)

    return retriever, chunks, encoder


@pytest.mark.parametrize("k", [0, 1, 2, 100])
def test_search_ranks_unique_chunks_by_best_segment(
    semantic_retriever: tuple[SemanticRetriever, list[Chunk], Mock],
    k: int,
) -> None:
    """Rank original chunks by their maximum segment similarity."""
    retriever, chunks, encoder = semantic_retriever
    results = retriever.search("example question", k=k)

    expected_chunks = chunks[:k]
    expected_scores = [1.0, 0.8][:k]

    assert [hit.chunk for hit in results] == expected_chunks
    assert [hit.score for hit in results] == pytest.approx(expected_scores)

    if k == 0:
        encoder.encode_query.assert_not_called()
    else:
        encoder.encode_query.assert_called_once_with("example question")


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_search_skips_blank_questions(
    semantic_retriever: tuple[SemanticRetriever, list[Chunk], Mock],
    query: str,
) -> None:
    """Return no sources for blank questions without calling the encoder."""
    retriever, _, encoder = semantic_retriever

    assert retriever.search(query, k=5) == []
    encoder.encode_query.assert_not_called()


@pytest.mark.parametrize("invalid_k", [-1, 1.5, "5", True])
def test_search_rejects_invalid_k(
    semantic_retriever: tuple[SemanticRetriever, list[Chunk], Mock],
    invalid_k: object,
) -> None:
    """Validate malformed runtime values before encoding a question."""
    retriever, _, encoder = semantic_retriever

    with pytest.raises(ValueError, match="non-negative integer"):
        getattr(retriever, "search")("example question", k=invalid_k)

    encoder.encode_query.assert_not_called()


@pytest.mark.parametrize(
    ("query_vector", "positions", "scores"),
    [
        ([0.6, 0.8], [1, 0], [0.96, 0.8]),
        ([-1.0, 0.0], [0, 1], [0.0, -0.8]),
        ([-0.6, -0.8], [0, 1], [-0.6, -0.96]),
        ([1.0, 0.0], [0, 1], [1.0, 1.0]),
    ],
)
def test_search_preserves_ranking_sources_and_negative_scores(
    semantic_retriever: tuple[SemanticRetriever, list[Chunk], Mock],
    query_vector: list[float],
    positions: list[int],
    scores: list[float],
) -> None:
    """Keep negative hits and break similarity ties by chunk position."""
    retriever, chunks, encoder = semantic_retriever
    encoder.encode_query.return_value = torch.tensor(
        [query_vector], dtype=torch.float32,
    )
    if scores == [1.0, 1.0]:
        retriever.embeddings[2] = torch.tensor([1.0, 0.0])

    results = retriever.search("example question", k=5)

    assert [hit.chunk for hit in results] == [chunks[i] for i in positions]
    assert [hit.score for hit in results] == pytest.approx(scores)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_search_rejects_nonfinite_scores(
    semantic_retriever: tuple[SemanticRetriever, list[Chunk], Mock],
    value: float,
) -> None:
    """Reject invalid numerical scores instead of returning ranked sources."""
    retriever, _, encoder = semantic_retriever
    encoder.encode_query.return_value = torch.tensor(
        [[value, 0.0]], dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="non-finite"):
        retriever.search("example question")


def test_search_propagates_question_budget_error(
    semantic_retriever: tuple[SemanticRetriever, list[Chunk], Mock],
) -> None:
    """Preserve the encoder's explicit rejection of an oversized question."""
    retriever, _, encoder = semantic_retriever
    encoder.encode_query.side_effect = ValueError(
        "Question exceeds the token budget"
    )

    with pytest.raises(ValueError, match="exceeds the token budget"):
        retriever.search("oversized question")


def test_retriever_rejects_encoder_dimension_mismatch(
    semantic_sample: tuple[Path, list[Chunk], Mock],
) -> None:
    """Reject an encoder incompatible with the persisted vector dimensions."""
    target, _, encoder = semantic_sample
    encoder.dimension = 3

    with patch("src.semantic_retriever.SemanticEncoder", return_value=encoder):
        with pytest.raises(ValueError, match="dimension does not match"):
            SemanticRetriever(target)


def test_retriever_rejects_empty_index(tmp_path: Path) -> None:
    """Reject an empty semantic index before constructing its encoder."""
    data = SemanticIndexData(
        metadata=SemanticIndexMetaData(
            encoder_config=EmbeddingConfig(),
            dimension=2,
            chunk_count=0,
            segment_count=0,
        ),
        chunks=[],
        segments=[],
    )
    target = tmp_path / "empty.pt"
    save_semantic_index(data, torch.empty((0, 2)), target)

    with patch("src.semantic_retriever.SemanticEncoder") as constructor:
        with pytest.raises(ValueError, match="empty semantic index"):
            SemanticRetriever(target)

    constructor.assert_not_called()
