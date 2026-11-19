"""Inspect semantic segments prepared from saved corpus chunks."""

from pathlib import Path

from src.embeddings import SemanticEncoder
from src.indexer import load_index


def main() -> None:
    """Segment a small sample while preserving chunk associations."""
    index = load_index(Path("data/processed"))
    chunks = index.chunks[:3]

    print("Loading encoder...", flush=True)
    encoder = SemanticEncoder()

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


if __name__ == "__main__":
    main()
