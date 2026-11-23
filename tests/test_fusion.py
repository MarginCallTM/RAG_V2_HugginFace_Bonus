"""Test reciprocal rank fusion with controlled sources and no models."""

from typing import cast

import pytest

from src.fusion import reciprocal_rank_fusion
from src.models import Chunk, ScoredChunk


def make_hit(name: str, score: float = 0.0, start: int = 100) -> ScoredChunk:
    """Create a source with explicit text, path and character bounds."""
    return ScoredChunk(
        chunk=Chunk(
            file_path=f"data/raw/{name}.txt",
            first_character_index=start,
            last_character_index=start + len(name),
            text=name,
        ),
        score=score,
    )


@pytest.mark.parametrize("k", [0, 1, 2, 100])
def test_fusion_combines_ranks_and_preserves_sources(k: int) -> None:
    """Fuse A/B/C and B/D/A using ranks rather than original scores."""
    a, b, c, d = [make_hit(name) for name in "ABCD"]
    lexical = [a.model_copy(update={"score": 999.0}), b, c]
    semantic = [b.model_copy(update={"score": -0.5}), d, a]

    results = reciprocal_rank_fusion([lexical, semantic], k=k)

    expected_chunks = [b.chunk, a.chunk, d.chunk, c.chunk][:k]
    assert [hit.chunk for hit in results] == expected_chunks
    assert [hit.score for hit in results] == pytest.approx(
        [1 / 62 + 1 / 61, 1 / 61 + 1 / 63, 1 / 62, 1 / 63][:k]
    )
    assert lexical[0].score == 999.0
    assert semantic[0].score == -0.5


@pytest.mark.parametrize("rankings", [[], [[]], [[], []]])
def test_fusion_handles_empty_rankings(
    rankings: list[list[ScoredChunk]],
) -> None:
    """Return no results when no method provides any candidates."""
    assert reciprocal_rank_fusion(rankings) == []


def test_fusion_handles_an_empty_first_method() -> None:
    """Use later rankings even when the first method finds nothing."""
    a = make_hit("A")

    results = reciprocal_rank_fusion([[], [a]])

    assert len(results) == 1
    assert results[0].chunk == a.chunk
    assert results[0].score == pytest.approx(1 / 61)


def test_fusion_counts_a_source_once_per_method() -> None:
    """Ignore repeated source entries but accumulate independent methods."""
    a, b = make_hit("A"), make_hit("B")

    results = reciprocal_rank_fusion([[a, a, b], [a]])

    assert [hit.chunk for hit in results] == [a.chunk, b.chunk]
    assert [hit.score for hit in results] == pytest.approx(
        [2 / 61, 1 / 63]
    )


def test_fusion_keeps_distinct_ranges_in_the_same_file() -> None:
    """Deduplicate by full source identity rather than just file path."""
    first, second = make_hit("A", start=100), make_hit("A", start=200)

    results = reciprocal_rank_fusion([[first], [second]])

    assert [hit.chunk for hit in results] == [first.chunk, second.chunk]
    assert [hit.score for hit in results] == pytest.approx([1 / 61, 1 / 61])


def test_fusion_breaks_ties_by_source_path() -> None:
    """Use deterministic source ordering rather than ranking arrival order."""
    a, b = make_hit("A"), make_hit("B")

    results = reciprocal_rank_fusion([[b], [a]])

    assert [hit.chunk for hit in results] == [a.chunk, b.chunk]


def test_fusion_accepts_zero_rank_constant() -> None:
    """Start ranks at one even when the smoothing constant is zero."""
    a, b = make_hit("A"), make_hit("B")

    results = reciprocal_rank_fusion([[a, b]], rank_constant=0)

    assert [hit.score for hit in results] == pytest.approx([1.0, 0.5])


@pytest.mark.parametrize("parameter", ["k", "rank_constant"])
@pytest.mark.parametrize("invalid", [-1, 1.5, "5", True])
def test_fusion_rejects_invalid_parameters(
    parameter: str,
    invalid: object,
) -> None:
    """Reject invalid runtime types and negative values explicitly."""
    with pytest.raises(ValueError, match=parameter):
        if parameter == "k":
            reciprocal_rank_fusion([], k=cast(int, invalid))
        else:
            reciprocal_rank_fusion([], rank_constant=cast(int, invalid))
