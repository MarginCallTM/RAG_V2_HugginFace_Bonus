"""Explore when semantic vectors require reconstruction."""

from src.embeddings import EmbeddingConfig
from src.semantic_incremental import requires_semantic_rebuild
from src.semantic_index import SemanticIndexMetaData


def make_metadata(
    config: EmbeddingConfig,
) -> SemanticIndexMetaData:
    """Create sample metadata without loading the embedding model"""
    return SemanticIndexMetaData(
        encoder_config=config,
        dimension=384,
        chunk_count=3,
        segment_count=6,
    )


def main() -> None:
    """Compare identical settings and two configuration changes."""
    previous = make_metadata(EmbeddingConfig(batch_size=32))
    identical = make_metadata(EmbeddingConfig(batch_size=32))
    different_batch = make_metadata(EmbeddingConfig(batch_size=2))
    different_budget = make_metadata(
        EmbeddingConfig(max_tokens=128),
    )

    print(
        "Same settings:",
        requires_semantic_rebuild(previous, identical),
    )
    print(
        "Different batch size:",
        requires_semantic_rebuild(previous, different_batch),
    )
    print(
        "Different token budget:",
        requires_semantic_rebuild(previous, different_budget),
    )


if __name__ == "__main__":
    main()
