"""Observe chunk-index persistence on disk"""

from pathlib import Path

from src.indexer import save_index, load_index
from src.models import Chunk, ChunkIndex


def main() -> None:
    """Save a small index and inspect the resulting file."""
    index = ChunkIndex(
        max_chunk_size=10,
        chunks=[
            Chunk(
                file_path="example.txt",
                first_character_index=0,
                last_character_index=7,
                text="Bonjour",
            )
        ],
    )

    processed_dir = Path("data/processed/demo")
    target = save_index(index, processed_dir)

    print("FICHIER FINAL :", target)
    print("EXISTE :", target.is_file())
    print("TAILLE SUR LE DISQUE :", target.stat().st_size, "octets")

    temporary = processed_dir / "chunks.json.tmp"
    print("FICHIER TEMPORAIRE PRESENT :", temporary.exists())

    with target.open(encoding="utf-8") as handle:
        serialized = handle.read()

    print("\nCONTENU SAUVEGARDE :")
    print(serialized)

    restored = load_index(processed_dir)

    print("\nTYPE RECHARGE :", type(restored).__name__)
    print("DONNEES IDENTIQUES :", restored == index)
    print("TEXTE DU PREMIER CHUNK :", restored.chunks[0].text)


if __name__ == "__main__":
    main()
