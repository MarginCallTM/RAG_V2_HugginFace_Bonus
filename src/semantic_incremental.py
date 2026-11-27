"""Reuse semantic segments during incremental index updates."""

import torch

from src.embeddings import TokenSegment, SemanticEncoder
from src.semantic_index import (
    SemanticIndexMetaData,
    SemanticIndexData,
    validate_embeddings,
)
from src.models import ChunkIndex


def requires_semantic_rebuild(
    previous: SemanticIndexMetaData,
    current: SemanticIndexMetaData,
) -> bool:
    """Detect settings that prevent reuse of semantic vectors."""
    previous_config = previous.encoder_config
    current_config = current.encoder_config

    return (
        previous.format_version != current.format_version
        or previous.preparation_version != current.preparation_version
        or previous_config.model_name != current_config.model_name
        or previous_config.revision != current_config.revision
        or previous_config.max_tokens != current_config.max_tokens
        or previous.dimension != current.dimension
        or previous.pooling != current.pooling
        or previous.normalization != current.normalization
        or previous.segmentation != current.segmentation
        or previous.dtype != current.dtype
    )


def remap_reusable_segments(
    previous_segments: list[TokenSegment],
    old_to_new: dict[int, int],
) -> tuple[list[TokenSegment], list[int]]:
    """Remap retianed segments and identify their previous vector rows."""
    current_segments: list[TokenSegment] = []
    retained_rows: list[int] = []

    for row, segment in enumerate(previous_segments):
        if segment.chunk_position not in old_to_new:
            continue

        current_segments.append(
            TokenSegment(
                chunk_position=old_to_new[segment.chunk_position],
                segment_position=segment.segment_position,
                input_ids=segment.input_ids,
            )
        )
        retained_rows.append(row)

    return current_segments, retained_rows


def encode_new_segments(
    current_index: ChunkIndex,
    old_to_new: dict[int, int],
    encoder: SemanticEncoder,
) -> tuple[list[TokenSegment], torch.Tensor]:
    """Encode non reusable chunks and restore global chunk positions."""
    reusable_positions = set(old_to_new.values())

    new_positions = [
        position
        for position in range(len(current_index.chunks))
        if position not in reusable_positions
    ]

    if not new_positions:
        return [], torch.empty(
            (0, encoder.dimension),
            dtype=torch.float32,
            device="cpu",
        )

    chunks_to_encode = [
        current_index.chunks[position]
        for position in new_positions
    ]

    local_segments = encoder.segment_chunks(chunks_to_encode)
    embeddings = encoder.encode_segments(local_segments)

    global_segments = [
        TokenSegment(
            chunk_position=new_positions[segment.chunk_position],
            segment_position=segment.segment_position,
            input_ids=segment.input_ids,
        )
        for segment in local_segments
    ]

    return global_segments, embeddings


def merge_segment_vectors(
    retained_segments: list[TokenSegment],
    retained_vectors: torch.Tensor,
    new_segments: list[TokenSegment],
    new_vectors: torch.Tensor,
) -> tuple[list[TokenSegment], torch.Tensor]:
    """Merge and order segments while preserving vector alignement."""
    groups = (
        (retained_segments, retained_vectors),
        (new_segments, new_vectors),
    )

    for segments, vectors in groups:
        if vectors.ndim != 2:
            raise ValueError("Expected a two-dimensional vector tensor")

        if len(segments) != vectors.shape[0]:
            raise ValueError("Segment and vector counts do not match")

    if retained_vectors.shape[1] != new_vectors.shape[1]:
        raise ValueError("Vector dimensions do not match")

    combined_segments = retained_segments + new_segments
    combined_vectors = torch.cat(
        (retained_vectors, new_vectors),
        dim=0,
    )

    ordered_rows = sorted(
        range(len(combined_segments)),
        key=lambda row: (
            combined_segments[row].chunk_position,
            combined_segments[row].segment_position,
        ),
    )

    ordered_segments = [
        combined_segments[row]
        for row in ordered_rows
    ]

    return ordered_segments, combined_vectors[ordered_rows]


def update_semantic_index(
    previous_data: SemanticIndexData,
    previous_vectors: torch.Tensor,
    current_index: ChunkIndex,
    old_to_new: dict[int, int],
    encoder: SemanticEncoder,
) -> tuple[SemanticIndexData, torch.Tensor]:
    """Build a validated semantic update without writing to disk."""
    if not current_index.chunks:
        raise ValueError("Cannot encode an empty chunk index")

    validate_embeddings(previous_data, previous_vectors)

    metadata = SemanticIndexMetaData(
        encoder_config=encoder.config,
        dimension=encoder.dimension,
        chunk_count=len(current_index.chunks),
        segment_count=0,
    )

    if requires_semantic_rebuild(previous_data.metadata, metadata):
        reusable_mapping: dict[int, int] = {}
    else:
        reusable_mapping = old_to_new

    retained_segments, retained_rows = remap_reusable_segments(
        previous_data.segments,
        reusable_mapping,
    )

    if retained_rows:
        retained_vectors = previous_vectors[retained_rows]
    else:
        retained_vectors = torch.empty(
            (0, encoder.dimension),
            dtype=torch.float32,
            device="cpu",
        )

    new_segments, new_vectors = encode_new_segments(
        current_index,
        reusable_mapping,
        encoder,
    )

    segments, embeddings = merge_segment_vectors(
        retained_segments,
        retained_vectors,
        new_segments,
        new_vectors,
    )

    metadata.segment_count = len(segments)

    data = SemanticIndexData(
        metadata=metadata,
        chunks=current_index.chunks,
        segments=segments,
    )

    validate_embeddings(data, embeddings)

    return data, embeddings
