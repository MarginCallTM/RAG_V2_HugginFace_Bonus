"""Test semantic index validation and persistence without loading MiniLM."""

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from src.embeddings import EmbeddingConfig, TokenSegment
from src.models import Chunk
from src.semantic_index import (
    SemanticIndexData,
    SemanticIndexMetaData,
    load_semantic_index,
    save_semantic_index,
    validate_embeddings,
)


@pytest.fixture
def semantic_data() -> SemanticIndexData:
    """Build one source chunk associated with two artificial segments."""
    text = "alpha beta"

    chunk = Chunk(
        file_path="data/raw/example.txt",
        first_character_index=100,
        last_character_index=100 + len(text),
        text=text,
    )

    segments = [
        TokenSegment(
            chunk_position=0,
            segment_position=position,
            input_ids=[101, position + 1, 102],
        )
        for position in range(2)
    ]

    metadata = SemanticIndexMetaData(
        encoder_config=EmbeddingConfig(max_tokens=8),
        dimension=2,
        chunk_count=1,
        segment_count=2,
    )

    return SemanticIndexData(
        metadata=metadata,
        chunks=[chunk],
        segments=segments,
    )


@pytest.fixture
def embeddings() -> torch.Tensor:
    """Provide two distinct normalized vectors in segment order."""
    return torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )


def test_validate_embeddings_accepts_valid_vectors(
    semantic_data: SemanticIndexData,
    embeddings: torch.Tensor,
) -> None:
    """Accept finite normalized float32 vectors matching the index."""
    validate_embeddings(semantic_data, embeddings)


def test_validate_embeddings_rejects_zero_vector(
    semantic_data: SemanticIndexData,
    embeddings: torch.Tensor,
) -> None:
    """Reject a zero vector even when the tensor shape is correct."""
    embeddings[0] = 0

    with pytest.raises(ValueError, match="unit L2 norms"):
        validate_embeddings(semantic_data, embeddings)


def test_semantic_data_rejects_inconsistent_segment_count(
    semantic_data: SemanticIndexData,
) -> None:
    """Reject metadata announcing more segments than are stored."""
    payload = semantic_data.model_dump(mode="json")
    payload["metadata"]["segment_count"] = 3

    with pytest.raises(ValidationError, match="Segment count"):
        SemanticIndexData.model_validate(payload)


def test_semantic_index_round_trip(
    tmp_path: Path,
    semantic_data: SemanticIndexData,
    embeddings: torch.Tensor,
) -> None:
    """Preserve source data, segment associations and exact vectors."""
    target = tmp_path / "nested" / "semantic_index.pt"

    saved_path = save_semantic_index(
        semantic_data,
        embeddings,
        target,
    )
    restored_data, restored_embeddings = load_semantic_index(saved_path)

    assert saved_path == target
    assert target.is_file()
    assert not target.with_name(target.name + ".tmp").exists()
    assert restored_data == semantic_data
    assert torch.equal(restored_embeddings, embeddings)
