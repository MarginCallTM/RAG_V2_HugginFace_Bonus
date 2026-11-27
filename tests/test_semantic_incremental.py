"""Verify semantic reuse during incremental updates."""

from unittest.mock import create_autospec

import pytest
import torch
from pydantic import ValidationError

from src.embeddings import EmbeddingConfig, SemanticEncoder, TokenSegment
from src.models import Chunk, ChunkIndex
from src.semantic_incremental import (
    encode_new_segments,
    merge_segment_vectors,
    remap_reusable_segments,
    requires_semantic_rebuild,
)
from src.semantic_index import SemanticIndexMetaData


def make_metadata(config: EmbeddingConfig) -> SemanticIndexMetaData:
    """Create valid metadata without loading a model."""
    return SemanticIndexMetaData(
        encoder_config=config,
        dimension=384,
        chunk_count=3,
        segment_count=6,
    )


def test_merge_keeps_vectors_aligned_during_sort() -> None:
    """Sort by chunk then segment without changing vector associations."""
    retained = [
        TokenSegment(
            chunk_position=chunk,
            segment_position=segment,
            input_ids=[101, token, 102],
        )
        for chunk, segment, token in [(2, 0, 20), (0, 1, 11), (0, 0, 10)]
    ]
    added = [TokenSegment(
        chunk_position=1, segment_position=0, input_ids=[101, 15, 102],
    )]
    retained_vectors = torch.tensor([[20., 0.], [11., 0.], [10., 0.]])
    new_vectors = torch.tensor([[15., 0.]])
    retained_before = retained_vectors.clone()
    new_before = new_vectors.clone()
    segments_before = [segment.model_dump() for segment in retained + added]

    segments, vectors = merge_segment_vectors(
        retained, retained_vectors, added, new_vectors,
    )

    assert [(s.chunk_position, s.segment_position) for s in segments] == [
        (0, 0), (0, 1), (1, 0), (2, 0),
    ]
    assert [segment.input_ids[1] for segment in segments] == [10, 11, 15, 20]
    assert torch.equal(vectors, torch.tensor([
        [10., 0.], [11., 0.], [15., 0.], [20., 0.],
    ]))
    assert torch.equal(retained_vectors, retained_before)
    assert torch.equal(new_vectors, new_before)
    assert [s.model_dump() for s in retained + added] == segments_before


@pytest.mark.parametrize("retained_count,new_count", [(1, 0), (0, 1), (0, 0)])
def test_merge_accepts_empty_groups(
    retained_count: int, new_count: int,
) -> None:
    """Keep the matrix shape when either or both groups are empty."""
    segment = TokenSegment(
        chunk_position=0, segment_position=0, input_ids=[101, 102],
    )
    segments, vectors = merge_segment_vectors(
        [segment] * retained_count,
        torch.ones((retained_count, 3)),
        [segment] * new_count,
        torch.ones((new_count, 3)),
    )

    assert segments == [segment] * (retained_count + new_count)
    assert torch.equal(vectors, torch.ones((retained_count + new_count, 3)))


@pytest.mark.parametrize(
    "retained_vectors,new_vectors,message",
    [
        (torch.ones(3), torch.empty((0, 3)), "two-dimensional"),
        (torch.empty((0, 3)), torch.ones((1, 1, 3)), "two-dimensional"),
        (torch.ones((1, 3)), torch.empty((0, 3)), "counts do not match"),
        (torch.empty((0, 3)), torch.ones((1, 3)), "counts do not match"),
        (torch.empty((0, 3)), torch.empty((0, 2)), "dimensions do not match"),
    ],
)
def test_merge_rejects_invalid_shapes(
    retained_vectors: torch.Tensor,
    new_vectors: torch.Tensor,
    message: str,
) -> None:
    """Reject malformed groups before assembling their rows."""
    with pytest.raises(ValueError, match=message):
        merge_segment_vectors([], retained_vectors, [], new_vectors)


def test_identical_semantic_settings_allow_reuse() -> None:
    """Accept separately constructed but equivalent settings."""
    previous = make_metadata(EmbeddingConfig())
    current = make_metadata(EmbeddingConfig())

    assert not requires_semantic_rebuild(previous, current)


@pytest.mark.parametrize(
    "config",
    [
        EmbeddingConfig(model_name="another/model"),
        EmbeddingConfig(revision="another-revision"),
        EmbeddingConfig(max_tokens=128),
    ],
    ids=["model", "revision", "token-budget"],
)
def test_changed_encoder_settings_require_rebuild(
    config: EmbeddingConfig,
) -> None:
    """Reject reuse when the encoding recipe changes."""
    previous = make_metadata(EmbeddingConfig())
    current = make_metadata(config)

    assert requires_semantic_rebuild(previous, current)
    assert requires_semantic_rebuild(current, previous)


def test_changed_dimension_requires_rebuild() -> None:
    """Reject vectors belonging to a different dimensional space."""
    previous = make_metadata(EmbeddingConfig())
    current = SemanticIndexMetaData(
        encoder_config=EmbeddingConfig(),
        dimension=768,
        chunk_count=3,
        segment_count=6,
    )

    assert requires_semantic_rebuild(previous, current)


def test_changed_batch_size_allows_reuse() -> None:
    """Batch scheduling does not invalidate existing embeddings."""
    previous = make_metadata(EmbeddingConfig(batch_size=32))
    current = make_metadata(EmbeddingConfig(batch_size=2))

    assert not requires_semantic_rebuild(previous, current)


def test_changed_corpus_counts_allow_reuse() -> None:
    """Corpus growth does not invalidate unchanged segment vectors."""
    previous = make_metadata(EmbeddingConfig())
    current = SemanticIndexMetaData(
        encoder_config=EmbeddingConfig(),
        dimension=384,
        chunk_count=5,
        segment_count=10,
    )

    assert not requires_semantic_rebuild(previous, current)


def test_unknown_preparation_version_is_rejected() -> None:
    """Reject unsupported preparation recipes during metadata loading."""
    payload = make_metadata(EmbeddingConfig()).model_dump()
    payload["preparation_version"] = 2

    with pytest.raises(ValidationError, match="preparation_version"):
        SemanticIndexMetaData.model_validate(payload)


def test_remap_preserves_segments_and_vector_alignment() -> None:
    """Keep both segments of a retained chunk after a deletion."""
    previous_segments = [
        TokenSegment(
            chunk_position=0,
            segment_position=0,
            input_ids=[101, 200, 102],
        ),
        TokenSegment(
            chunk_position=1,
            segment_position=0,
            input_ids=[101, 300, 102],
        ),
        TokenSegment(
            chunk_position=1,
            segment_position=1,
            input_ids=[101, 400, 102],
        ),
    ]

    previous_vectors = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    original_segments = [
        segment.model_dump()
        for segment in previous_segments
    ]

    segments, retained_rows = remap_reusable_segments(
        previous_segments,
        old_to_new={1: 0},
    )

    assert retained_rows == [1, 2]
    assert [segment.chunk_position for segment in segments] == [0, 0]
    assert [segment.segment_position for segment in segments] == [0, 1]
    assert [segment.input_ids for segment in segments] == [
        [101, 300, 102],
        [101, 400, 102],
    ]

    retained_vectors = previous_vectors[retained_rows]

    assert torch.equal(
        retained_vectors,
        torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]),
    )

    assert [
        segment.model_dump()
        for segment in previous_segments
    ] == original_segments


@pytest.mark.parametrize("reuse", [True, False], ids=["partial", "none"])
def test_encode_new_segments_restores_global_positions(reuse: bool) -> None:
    """Encode only selected chunks and preserve segment/vector order."""
    chunks = [
        Chunk(
            file_path=f"file_{position}.txt",
            first_character_index=0,
            last_character_index=5,
            text="Hello",
        )
        for position in range(4)
    ]
    index = ChunkIndex(max_chunk_size=2000, chunks=chunks)
    old_to_new = {4: 0, 5: 2} if reuse else {}
    positions = [1, 3] if reuse else [0, 1, 2, 3]

    local_segments = [
        TokenSegment(
            chunk_position=local_position,
            segment_position=segment_position,
            input_ids=[101, 200 + local_position, 102],
        )
        for local_position in range(len(positions))
        for segment_position in range(2)
    ]
    original_segments = [segment.model_dump() for segment in local_segments]
    vectors = torch.eye(len(local_segments), dtype=torch.float32)
    encoder = create_autospec(SemanticEncoder, instance=True)
    encoder.segment_chunks.return_value = local_segments
    encoder.encode_segments.return_value = vectors

    segments, embeddings = encode_new_segments(index, old_to_new, encoder)

    encoder.segment_chunks.assert_called_once_with(
        [chunks[position] for position in positions],
    )
    encoder.encode_segments.assert_called_once_with(local_segments)
    assert [segment.chunk_position for segment in segments] == [
        position for position in positions for _ in range(2)
    ]
    assert [segment.segment_position for segment in segments] == [
        0, 1,
    ] * len(positions)
    assert [segment.input_ids for segment in segments] == [
        segment.input_ids for segment in local_segments
    ]
    assert embeddings is vectors
    assert [segment.model_dump() for segment in local_segments] == (
        original_segments
    )


def test_encode_new_segments_skips_encoding_when_all_reusable() -> None:
    """Return a correctly shaped empty tensor without encoder calls."""
    chunk = Chunk(
        file_path="keep.txt",
        first_character_index=0,
        last_character_index=5,
        text="Hello",
    )
    index = ChunkIndex(max_chunk_size=2000, chunks=[chunk])
    encoder = create_autospec(SemanticEncoder, instance=True)
    encoder.dimension = 384

    segments, embeddings = encode_new_segments(index, {2: 0}, encoder)

    encoder.segment_chunks.assert_not_called()
    encoder.encode_segments.assert_not_called()
    assert segments == []
    assert tuple(embeddings.shape) == (0, 384)
    assert embeddings.dtype == torch.float32
    assert embeddings.device.type == "cpu"
