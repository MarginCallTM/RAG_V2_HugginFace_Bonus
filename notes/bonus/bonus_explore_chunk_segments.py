"""Inspect semantic segments prepared from saved corpus chunks."""

import torch

from pathlib import Path

from src.embeddings import SemanticEncoder, EmbeddingConfig
from src.indexer import load_index
from src.semantic_index import (
    SemanticIndexData,
    SemanticIndexMetaData,
    validate_embeddings,
    save_semantic_index,
    load_semantic_index,
)


def main() -> None:
    """Segment a small sample while preserving chunk associations."""
    index = load_index(Path("data/processed"))
    chunks = index.chunks[:3]

    print("Loading encoder...", flush=True)
    encoder = SemanticEncoder(EmbeddingConfig(batch_size=2))

    print("Model:", encoder.config.model_name)
    print("Revision:", encoder.config.revision)

    segments = encoder.segment_chunks(chunks)

    print("Original chunks:", len(chunks))
    print("Token segments:", len(segments))

    for segment in segments:
        chunk = chunks[segment.chunk_position]

        print(
            f"Chunk {segment.chunk_position}, "
            f"segment {segment.segment_position}, "
            f"tokens={len(segment.input_ids)}"
        )
        print(
            f" Sources: {chunk.file_path} "
            f"[{chunk.first_character_index}:"
            f"{chunk.last_character_index}]"
        )

    print(
        "All segments within budget:",
        all(
            len(segment.input_ids) <= encoder.config.max_tokens
            for segment in segments
        ),
    )

    embeddings = encoder.encode_segments(segments)

    metadata = SemanticIndexMetaData(
        encoder_config=encoder.config,
        dimension=encoder.dimension,
        chunk_count=len(chunks),
        segment_count=len(segments)
    )

    semantic_data = SemanticIndexData(
        metadata=metadata,
        chunks=chunks,
        segments=segments,
    )

    invalid_embeddings = embeddings.clone()
    invalid_embeddings[0] = 0

    try:
        validate_embeddings(semantic_data, invalid_embeddings)
    except ValueError as error:
        print("\nInvalid vectors rejected:", error)
    else:
        raise AssertionError("A zero vector should have been rejected")

    validate_embeddings(semantic_data, embeddings)
    print("\nSemantic index data and vectors validated")

    target = Path(
        "data/output/experiments/semantic/sample/semantic_index.pt"
    )

    saved_path = save_semantic_index(
        semantic_data,
        embeddings,
        target,
    )

    print("Semantic index saved:", saved_path)
    print("File exists:", saved_path.is_file())
    print("File size in bytes:", saved_path.stat().st_size)

    print("\nSemantic index metadata:")
    print(metadata.model_dump_json(indent=2))

    restored_data, restored_embeddings = load_semantic_index(saved_path)

    print("\nSemantic index reloaded")
    print("Same index data:", restored_data == semantic_data)
    print(
        "Same vectors:",
        torch.equal(restored_embeddings, embeddings),
    )
    print("Restored shape:", tuple(restored_embeddings.shape))

    print("\nEmbeddings shape:", tuple(embeddings.shape))
    print("Embedding norms:", embeddings.norm(dim=1).tolist())
    print("Device:", embeddings.device)
    print("Dtype:", embeddings.dtype)

    for row, segment in enumerate(segments):
        print(
            f"Vector {row} -> chunk {segment.chunk_position}, "
            f"segment {segment.segment_position}"
        )


if __name__ == "__main__":
    main()
