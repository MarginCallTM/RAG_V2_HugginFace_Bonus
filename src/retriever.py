"""Prepare lexical retrieval over the saved corpus chunks."""

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.models import ChunkIndex, ScoredChunk
from src.tokenizer import tokenize


class Retriever:
    """Keep corpus chunks aligned with their BM25 representation"""

    def __init__(
        self,
        index: ChunkIndex,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Tokenize chunks and prepare reusable BM25 statistics"""
        if not index.chunks:
            raise ValueError("Cannot search an empty index")

        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1 must be positive and b between 0 and 1")

        self.chunks = index.chunks

        tokenized = [
            tokenize(chunk.text)
            for chunk in tqdm(
                self.chunks,
                desc="Tokenizing",
                unit="chunk",
            )
        ]

        if not any(tokenized):
            raise ValueError("Index contains no searchable terms")

        self.bm25 = BM25Okapi(
            tokenized,
            k1=k1,
            b=b,
        )

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        """Return up to k chunks with strictly positive BM25 scores"""
        if k <= 0:
            return []

        terms = sorted(set(tokenize(query)))

        if not terms:
            return []

        scores = self.bm25.get_scores(terms)

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True
        )

        return [
            ScoredChunk(
                chunk=self.chunks[position],
                score=float(score),
            )
            for position, score in ranked[:k]
            if score > 0
        ]
