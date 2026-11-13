"""Observe BM25 preparation over the saved corpus index."""

from pathlib import Path
from time import perf_counter

from src.indexer import load_index
from src.retriever import Retriever


def main() -> None:
    """Measure index loading and retriever preparation separately"""
    started = perf_counter()

    index = load_index(Path("data/processed"))
    loaded_at = perf_counter()

    retriever = Retriever(index)
    ready_at = perf_counter()

    print("\nCHUNKS :", len(retriever.chunks))
    print("CHUNKS COTE BM25 :", retriever.bm25.corpus_size)
    print("VOCABULAIRE :", len(retriever.bm25.idf))
    print("LONGUEUR MOYENNE :", retriever.bm25.avgdl)
    print("CHARGEMENT :", f"{loaded_at - started:.2f} s")
    print("PREPARATION BM25 :", f"{ready_at - loaded_at:.2f} s")

    query = (
        "What HTTP endpoint is used to dynamically load "
        "a LoRA adapter in vLLM?"
    )

    print("\nQUESTION :", query)

    for rank, result in enumerate(retriever.search(query, k=5), start=1):
        chunk = result.chunk

        print(f"\nRANG {rank} — SCORE {result.score:.3f}")
        print("SOURCE :", chunk.file_path)
        print(
            "BORNES :",
            chunk.first_character_index,
            chunk.last_character_index,
        )
        print(chunk.text[:500])


if __name__ == "__main__":
    main()
