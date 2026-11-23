"""Search real source chunks from a saved semantic sample index"""

from pathlib import Path

from src.semantic_retriever import SemanticRetriever


def main() -> None:
    """Inspect ranked source locations without re-encoding the corpus"""
    path = Path(
        "data/output/experiments/semantic/sample/semantic_index.pt"
    )

    print("Loading semantic retriever...", flush=True)
    retriever = SemanticRetriever(path)

    question = (
        "Which environment variable sets the maximum allowed wheel size?"
    )

    print("\nQuestion:", question)

    for rank, hit in enumerate(
        retriever.search(question, k=3),
        start=1,
    ):
        chunk = hit.chunk
        print(f"\n{rank}. cosine={hit.score:.4f}")
        print(
            f"{chunk.file_path} "
            f"[{chunk.first_character_index}:"
            f"{chunk.last_character_index}]"
        )
        print("Preview:", repr(chunk.text[:150]))


if __name__ == "__main__":
    main()
