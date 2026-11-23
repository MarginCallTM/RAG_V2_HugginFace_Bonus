"""Describe the semantic index and its preparation settings."""

import torch

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pathlib import Path
from src.embeddings import (
    EmbeddingConfig,
    TokenSegment,
    SemanticEncoder,
)
from src.models import Chunk, ChunkIndex


class SemanticIndexMetaData(BaseModel):
    """Record how semantic vectors were prepared and organised."""

    format_version: Literal[1] = 1
    preparation_version: Literal[1] = 1

    encoder_config: EmbeddingConfig

    dimension: int = Field(ge=1)
    chunk_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)

    pooling: Literal["masked_mean"] = "masked_mean"
    normalization: Literal["l2"] = "l2"
    segmentation: Literal["token_windows_no_overlap"] = (
        "token_windows_no_overlap"
    )
    dtype: Literal["float32"] = "float32"


class SemanticIndexData(BaseModel):
    """Keep semantic metadata, chunks and segment associations together."""

    metadata: SemanticIndexMetaData
    chunks: list[Chunk]
    segments: list[TokenSegment]

    @model_validator(mode="after")
    def check_consistency(self) -> "SemanticIndexData":
        """Validate counts, chunk references and segment numbering."""
        if len(self.chunks) != self.metadata.chunk_count:
            raise ValueError("Chunk count does not match metadata")

        if len(self.segments) != self.metadata.segment_count:
            raise ValueError("Segment count does not match metadata")

        next_positions = [0] * len(self.chunks)
        token_budget = self.metadata.encoder_config.max_tokens

        for segment in self.segments:
            if segment.chunk_position >= len(self.chunks):
                raise ValueError("Segment refers to an unknow chunk")

            if len(segment.input_ids) > token_budget:
                raise ValueError("Segment exceeds the token budget")

            expected = next_positions[segment.chunk_position]

            if segment.segment_position != expected:
                raise ValueError("Invalid segment numbering for a chunk")

            next_positions[segment.chunk_position] += 1
        if any(count == 0 for count in next_positions):
            raise ValueError("Each chunk must have at least one segment")

        return self


def validate_embeddings(
    data: SemanticIndexData,
    embeddings: torch.Tensor,
) -> None:
    """Validate the vector tensor against the semantic index metadata"""
    expected_shape = (
        data.metadata.segment_count,
        data.metadata.dimension,
    )

    if tuple(embeddings.shape) != expected_shape:
        raise ValueError(
            f"Expected embeddings shape {expected_shape}, "
            f"got {tuple(embeddings.shape)}"
        )

    if embeddings.device.type != "cpu":
        raise ValueError("Embeddings must be stored on CPU")

    if embeddings.dtype != torch.float32:
        raise ValueError("Embeddings must use float32")

    if not torch.isfinite(embeddings).all().item():
        raise ValueError("Embeddings contain non-finite values")

    norms = embeddings.norm(p=2, dim=1)

    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=1e-4,
        atol=1e-5,
    ):
        raise ValueError("Embeddings must have unit L2 norms")


def build_semantic_index(
    index: ChunkIndex,
    config: EmbeddingConfig | None = None,
) -> tuple[SemanticIndexData, torch.Tensor]:
    """Build validated semantic data and vectors from corpus chunks.\

    Args:
        index: Existing corpus chunks with their source locations.
        config: Encoder settings, or defaults when omitted.

    Returns:
        Semantic index data and aligned normalized CPU vectors.

    Raise:
        ValueError: If the index is emptu or generated data is invalid.
    """
    if not index.chunks:
        raise ValueError("Cannot encode an empty chunk index")

    encoder = SemanticEncoder(config)
    segments = encoder.segment_chunks(index.chunks)
    embeddings = encoder.encode_segments(segments)

    metadata = SemanticIndexMetaData(
        encoder_config=encoder.config,
        dimension=encoder.dimension,
        chunk_count=len(index.chunks),
        segment_count=len(segments),
    )

    data = SemanticIndexData(
        metadata=metadata,
        chunks=index.chunks,
        segments=segments,
    )
    validate_embeddings(data, embeddings)

    return data, embeddings


def save_semantic_index(
    data: SemanticIndexData,
    embeddings: torch.Tensor,
    target: Path,
) -> Path:
    """Validate and save semantic data and vectors in one local file."""
    serialized_data = data.model_dump(mode="json")
    SemanticIndexData.model_validate(serialized_data)
    validate_embeddings(data, embeddings)

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")

    torch.save(
        {
            "data": serialized_data,
            "embeddings": embeddings,
        },
        temporary,
    )

    temporary.replace(target)

    return target


def load_semantic_index(
    path: Path,
) -> tuple[SemanticIndexData, torch.Tensor]:
    """Load and validate semantic data and vectors from a local index."""
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(payload, dict):
        raise ValueError("Semantic index must contain a dictionary")

    if "data" not in payload or "embeddings" not in payload:
        raise ValueError("Semantic index is missing required fields")

    data = SemanticIndexData.model_validate(payload["data"])
    embeddings = payload["embeddings"]

    if not isinstance(embeddings, torch.Tensor):
        raise ValueError("Semantic index embeddinfs must be a tensor")

    validate_embeddings(data, embeddings)

    return data, embeddings
