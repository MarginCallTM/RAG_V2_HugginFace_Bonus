"""Observe chunk-index serialization and validation"""

from pydantic import ValidationError

from src.models import Chunk, ChunkIndex


def main() -> None:
    """Serialize a small index and reject an inconsistent size limit."""
    text = "hello my friend"

    chunk = Chunk(
        file_path="example.txt",
        first_character_index=0,
        last_character_index=len(text),
        text=text,
    )

    index = ChunkIndex(
        max_chunk_size=10,
        chunks=[chunk],
    )

    serialized = index.model_dump_json(indent=2)

    print("JSON :")
    print(serialized)

    restored = ChunkIndex.model_validate_json(serialized)

    print("\nTYPE RECHARGE :", type(restored).__name__)
    print("TYPE DU PREMIER CHUNK :", type(restored.chunks[0]).__name__)
    print("DONNEES IDENTIQUES :", restored == index)

    try:
        ChunkIndex(
            max_chunk_size=5,
            chunks=[chunk],
        )
    except ValidationError as error:
        print("\nINDEX INCOHERENT REFUSE :")
        print(error)


if __name__ == "__main__":
    main()
