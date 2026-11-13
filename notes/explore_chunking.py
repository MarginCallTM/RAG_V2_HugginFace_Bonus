"""Inspect Markdown chunks from the real LoRa documentation"""

from pathlib import Path

from src.chunking import split_markdown
from src.ingest import read_text


def main() -> None:
    """Check chunk invariants and display passafes mentioning loading"""
    path = Path("data/raw/vllm-0.10.1/docs/features/lora.md")
    text = read_text(path)

    for size in (2000, 500):
        chunks = split_markdown(
            file_path=path.as_posix(),
            text=text,
            max_chunk_size=size,
        )

        assert "".join(chunk.text for chunk in chunks) == text

        for chunk in chunks:
            start = chunk.first_character_index
            end = chunk.last_character_index

            assert chunk.text == text[start:end]
            assert 0 < len(chunk.text) <= size

        print(f"\nTAILLE MAXIMALE : {size}")
        print("NOMBRE DE CHUNKS :", len(chunks))

        for index, chunk in enumerate(chunks):
            if (
                "can you transcribe the speech into a written format?"
                in chunk.text
            ):
                print(
                    f"\nCHUNK {index} "
                    f"[{chunk.first_character_index}:"
                    f"{chunk.last_character_index}]"
                )
                print("LONGUEUR :", len(chunk.text))
                print(chunk.text)


if __name__ == "__main__":
    main()
