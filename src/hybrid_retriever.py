"""Combine lexical and semantic retrieval using reciprocal rank fusion."""

from src.fusion import reciprocal_rank_fusion
from src.models import ScoredChunk
from src.retrieval import SearchRetriever
from math import isfinite


class HybridRetriever:
    """Reuse two retrievers and fuse their ranked candidates."""

    def __init__(
            self,
            lexical: SearchRetriever,
            semantic: SearchRetriever,
            candidate_k: int = 20,
            rank_constant: int = 60,
            lexical_weight: float = 1.0,
            semantic_weight: float = 1.0,
    ) -> None:
        """Store retrievers and validate fusion settings.

        Args:
                lexical: Already initialized lexical retriever.
                semantic: Already initialized semantic retriever.
                candidate_k: Minimum candidate count requested per method.
                rank_constant: Non-negative RRF smoothing constant.
                lexical_weight: Positive finite weight for lexical ranks.
                semantic_weight: Positive finite weight for semantic ranks.
        """
        if type(candidate_k) is not int or candidate_k <= 0:
            raise ValueError("candidate_k must be a positive integer")

        if type(rank_constant) is not int or rank_constant < 0:
            raise ValueError(
                "rank_constant must be a non-negative integer"
            )

        for weight in (lexical_weight, semantic_weight):
            if not isfinite(weight) or weight <= 0:
                raise ValueError("Weights must be positive and finite")

        self.lexical = lexical
        self.semantic = semantic
        self.candidate_k = candidate_k
        self.rank_constant = rank_constant
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Retrieve candidates from both methods and return fused sources."""
        if type(k) is not int or k < 0:
            raise ValueError("k must be a non-negative integer")

        if k == 0 or not query.strip():
            return []

        candidate_count = max(k, self.candidate_k)

        lexical_hits = self.lexical.search(query, k=candidate_count)
        semantic_hits = self.semantic.search(query, k=candidate_count)

        return reciprocal_rank_fusion(
            [lexical_hits, semantic_hits],
            k=k,
            rank_constant=self.rank_constant,
            weights=[self.lexical_weight, self.semantic_weight],
        )
