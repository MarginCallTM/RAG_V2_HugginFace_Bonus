"""Select a retrieval method through a shared search interface."""

from pathlib import Path
from typing import Protocol

from src.models import ScoredChunk


class SearchRetriever(Protocol):
    """Describe the search interface shared by retrieval methods."""

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Return ranked chunks with their scores."""
        ...


def load_retriever(
        method: str,
        processed_dir: Path,
        semantic_index_path: Path,
) -> SearchRetriever:
    """Load the requested retriever without building a new index"""
    if method == "bm25":
        from src.indexer import load_index
        from src.retriever import Retriever

        return Retriever(load_index(processed_dir))

    if method == "semantic":
        from src.semantic_retriever import SemanticRetriever

        return SemanticRetriever(semantic_index_path)

    raise ValueError("Method must be bm25 or semantic")
