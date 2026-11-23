"""Combine ranked source chunks using Reciprocal Rank Fusion."""

from src.models import Chunk, ScoredChunk
from math import isfinite

SourceKey = tuple[str, int, int]


def reciprocal_rank_fusion(
        rankings: list[list[ScoredChunk]],
        k: int = 5,
        rank_constant: int = 60,
        weights: list[float] | None = None,
) -> list[ScoredChunk]:
    """Fuse rankings using source identity and reciprocal ranks.

    Args:
            rankings: Lists of hits ordered from most to least relevant.
            k: Maximum number of unique sources returned.
            rank_constant: Non-negative constant smoothing rank contributions.
            weights: Positive finite weights aligned with rankings.

    Returns:
            Unique chunks ordered by decreasing RRF score.
            Ties are resolved by source Path and character bounds.

    Raises:
            ValueError: If k or rank_constant is invalid
    """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a non-negative integer")

    if type(rank_constant) is not int or rank_constant < 0:
        raise ValueError("rank_constant must be a non-negative integer")

    if weights is None:
        weights = [1.0] * len(rankings)

    if len(weights) != len(rankings):
        raise ValueError("Each ranking must have one weight")

    if any(not isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("Weights must be positive and finite")

    if k == 0:
        return []

    scores: dict[SourceKey, float] = {}
    chunks: dict[SourceKey, Chunk] = {}

    for ranking, weight in zip(rankings, weights):
        seen: set[SourceKey] = set()

        for rank, hit in enumerate(ranking, start=1):
            chunk = hit.chunk
            key = (
                chunk.file_path,
                chunk.first_character_index,
                chunk.last_character_index,
            )

            if key in seen:
                continue

            seen.add(key)

            if key not in chunks:
                chunks[key] = chunk
                scores[key] = 0.0

            scores[key] += weight / (rank_constant + rank)

    ordered_keys = sorted(
        scores,
        key=lambda key: (-scores[key], key),
    )

    return [
        ScoredChunk(
            chunk=chunks[key],
            score=scores[key],
        )
        for key in ordered_keys[:k]
    ]
