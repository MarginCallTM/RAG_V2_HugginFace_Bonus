"""Observe index construction before disk persistence."""

from pathlib import Path

from src.indexer import build_index


def main() -> None:
    """Build the corpus index and display its configuration and size"""
    project_root = Path.cwd()

    index = build_index(
        raw_dir=project_root / "data" / "raw" / "vllm-0.10.1",
        project_root=project_root,
        max_chunk_size=2000,
    )

    print("\nTYPE :", type(index).__name__)
    print("TAILLE MAXIMALE :", index.max_chunk_size)
    print("NOMBRE DE CHUNKS :", len(index.chunks))
    print("PREMIERE SOURCE :", index.chunks[0].file_path)


if __name__ == "__main__":
    main()
