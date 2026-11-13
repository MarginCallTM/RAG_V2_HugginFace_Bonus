"""Tests for source overlap and retrieval evaluation."""

import pytest

from src.evaluation import (
    source_iou,
    evaluate_recall,
    source_is_found,
    recall_at_k,
)
from src.models import (
    AnsweredDataset,
    AnsweredQuestion,
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (100, 200, 1.0),
        (100, 300, 0.5),
        (150, 250, 1 / 3),
        (120, 180, 0.6),
        (200, 300, 0.0),
        (300, 400, 0.0),
        (0, 100, 0.0),
    ],
)
def test_source_iou_compares_character_ranges(
    start: int,
    end: int,
    expected: float,
) -> None:
    """Measure overlap for identical, nested and separate ranges."""
    reference = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=start,
        last_character_index=end,
    )

    assert source_iou(reference, retrieved) == pytest.approx(expected)
    assert source_iou(retrieved, reference) == pytest.approx(expected)


def test_source_iou_rejects_different_paths() -> None:
    """Return zero for identical ranges in different files."""
    reference = MinimalSource(
        file_path="data/raw/first.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved = MinimalSource(
        file_path="data/raw/second.py",
        first_character_index=100,
        last_character_index=200,
    )

    assert source_iou(reference, retrieved) == 0.0


@pytest.mark.parametrize(
    ("file_path", "start", "end", "expected"),
    [
        ("data/raw/example.py", 100, 200, True),
        ("data/raw/example.py", 100, 300, True),
        ("data/raw/example.py", 199, 200, False),
        ("data/raw/example.py", 200, 300, False),
        ("data/raw/other.py", 100, 200, False),
    ],
)
def test_source_is_found_checks_path_and_overlap(
    file_path: str,
    start: int,
    end: int,
    expected: bool,
) -> None:
    """Require the same file and sufficient character overlap."""
    reference = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    retrieved = MinimalSource(
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
    )

    assert source_is_found(reference, [retrieved]) is expected


def test_source_is_found_accepts_a_later_match() -> None:
    """Find a reference even when the first result does not match."""
    reference = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    unrelated = MinimalSource(
        file_path="data/raw/other.py",
        first_character_index=100,
        last_character_index=200,
    )

    assert source_is_found(reference, [unrelated, reference]) is True


def test_source_is_found_handles_empty_results() -> None:
    """Report an unfound reference when no sources were retrieved."""
    reference = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )

    assert source_is_found(reference, []) is False


@pytest.mark.parametrize(
    ("retrieved_names", "k", "expected"),
    [
        ([], 5, 0.0),
        (["a"], 5, 0.5),
        (["a", "b"], 5, 1.0),
        (["a", "a", "a"], 5, 0.5),
        (["a", "b"], 1, 0.5),
        (["a", "b"], 0, 0.0),
        (["other", "a", "b"], 2, 0.5),
    ],
)
def test_recall_at_k_counts_reference_sources(
    retrieved_names: list[str],
    k: int,
    expected: float,
) -> None:
    """Count each reference once and consider only the top-k."""
    sources = {
        name: MinimalSource(
            file_path=f"data/raw/{name}.py",
            first_character_index=100,
            last_character_index=200,
        )
        for name in ("a", "b", "other")
    }
    references = [sources["a"], sources["b"]]
    retrieved = [sources[name] for name in retrieved_names]

    score = recall_at_k(references, retrieved, k=k)

    assert score == pytest.approx(expected)


def test_recall_at_k_rejects_missing_references() -> None:
    """Refuse to compute recall without reference sources."""
    with pytest.raises(ValueError, match="at least one reference"):
        recall_at_k([], [], k=5)


def test_recall_at_k_rejects_negative_k() -> None:
    """Reject a negative limit instead of using negative slicing."""
    reference = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )

    with pytest.raises(ValueError, match="non-negative integer"):
        recall_at_k([reference], [], k=-1)


@pytest.mark.parametrize(
    ("include_q3", "expected_mean"),
    [
        (True, 5 / 6),
        (False, 0.5),
    ],
)
def test_evaluate_recall_matches_ids_and_averages_questions(
    include_q3: bool,
    expected_mean: float,
) -> None:
    """Match reordered IDs and count missing questions as zero."""
    sources = {
        name: MinimalSource(
            file_path=f"data/raw/{name}.py",
            first_character_index=100,
            last_character_index=200,
        )
        for name in ("a", "b", "c", "d")
    }
    dataset = AnsweredDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id=question_id,
                question="Example question",
                sources=[sources[name] for name in names],
                answer="Reference answer",
            )
            for question_id, names in (
                ("q1", ["a"]),
                ("q2", ["b", "c"]),
                ("q3", ["d"]),
            )
        ]
    )
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id=question_id,
                question="Example question",
                retrieved_sources=[sources[name]],
            )
            for question_id, name in (
                ("q3", "d"),
                ("q1", "a"),
                ("q2", "b"),
            )
            if include_q3 or question_id != "q3"
        ],
        k=5,
    )

    scores, mean_recall = evaluate_recall(dataset, results, k=5)

    assert scores == {
        "q1": 1.0,
        "q2": 0.5,
        "q3": 1.0 if include_q3 else 0.0,
    }
    assert mean_recall == pytest.approx(expected_mean)


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("duplicate_reference", "Duplicate question_id"),
        ("duplicate_result", "Duplicate question_id"),
        ("unknown_id", "question_id in search results"),
        ("empty_dataset", "Evaluation requires"),
        ("missing_sources", "at least one reference source"),
        ("negative_k", "non-negative integer"),
        ("too_large_k", "available result limit"),
        ("negative_result_k", "available result limit"),
    ],
)
def test_evaluate_recall_rejects_invalid_inputs(
    case: str,
    expected_message: str,
) -> None:
    """Reject ambiguous inputs and undefined evaluation cases."""
    source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    question = AnsweredQuestion(
        question_id="q1",
        question="Example question",
        sources=[source],
        answer="Reference answer",
    )
    result = MinimalSearchResults(
        question_id="q1",
        question="Example question",
        retrieved_sources=[source],
    )
    dataset = AnsweredDataset(rag_questions=[question])
    results = StudentSearchResults(search_results=[result], k=5)
    k = 5

    if case == "duplicate_reference":
        dataset.rag_questions.append(question)
    elif case == "duplicate_result":
        results.search_results.append(result)
    elif case == "unknown_id":
        result.question_id = "unknown"
    elif case == "empty_dataset":
        dataset.rag_questions.clear()
    elif case == "missing_sources":
        question.sources.clear()
    elif case == "negative_k":
        k = -1
    elif case == "too_large_k":
        k = 6
    elif case == "negative_result_k":
        results.k = -1

    with pytest.raises(ValueError, match=expected_message):
        evaluate_recall(dataset, results, k=k)
