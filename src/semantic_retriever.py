"""Retrieve source chunks using semantic segment similarities"""

from pathlib import Path

import torch

from src.embeddings import SemanticEncoder
from src.models import ScoredChunk
from src.semantic_index import load_semantic_index


class SemanticRetriever:
    """Reuse a saved semantic index and its configured CPU encoder"""

    def __init__(self, index_path: Path) -> None:
        """Load source data, segment vectors and the matching encoder."""
        self.data, self.embeddings = load_semantic_index(index_path)

        if not self.data.chunks:
            raise ValueError("Cannot search an empty semantic index")

        if any(len(chunk.text) > 2000 for chunk in self.data.chunks):
            raise ValueError("A source chunk exceeds 2000 characters")

        self.encoder = SemanticEncoder(self.data.metadata.encoder_config)

        if self.encoder.dimension != self.data.metadata.dimension:
            raise ValueError("Encoder dimension does not match the index")

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Return unique chunks ranked by their best segment similarity"""
        if type(k) is not int or k < 0:
            raise ValueError("k must be a non-negative integer")

        if k == 0 or not query.strip():
            return []

        query_embedding = self.encoder.encode_query(query)
        scores = query_embedding @ self.embeddings.T

        if not torch.isfinite(scores).all().item():
            raise ValueError("Semantic search produced non-finite scores")

        best_scores: dict[int, float] = {}

        for row, score in enumerate(scores[0].tolist()):
            chunk_position = self.data.segments[row].chunk_position

            if chunk_position not in best_scores:
                best_scores[chunk_position] = score
            elif score > best_scores[chunk_position]:
                best_scores[chunk_position] = score

        ranked = sorted(
            best_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )

        return [
            ScoredChunk(
                chunk=self.data.chunks[position],
                score=score,
            )
            for position, score in ranked[:k]
        ]
