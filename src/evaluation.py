"""Evalutate retrieved sources against reference sources."""

from src.models import (
    AnsweredDataset,
    MinimalSource,
    StudentSearchResults,
)


def source_iou(
        reference: MinimalSource,
        retrieved: MinimalSource,
) -> float:
    """Return character-range IoU, or zero for different files."""
    if reference.file_path != retrieved.file_path:
        return 0.0

    intersection_start = max(
        reference.first_character_index,
        retrieved.first_character_index,
    )
    intersection_end = min(
        reference.last_character_index,
        retrieved.last_character_index,
    )
    intersection = max(0, intersection_end - intersection_start)

    reference_lenght = (
        reference.last_character_index
        - reference.first_character_index
    )
    retrieved_lenght = (
        retrieved.last_character_index
        - retrieved.first_character_index
    )
    union = reference_lenght + retrieved_lenght - intersection

    return intersection / union


def source_is_found(
    reference: MinimalSource,
    retrieved_sources: list[MinimalSource],
) -> bool:
    """Check whether any retrieved source matches the reference."""
    return any(
        source_iou(reference, retrieved) > 0.05
        for retrieved in retrieved_sources
    )


def recall_at_k(
    reference_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    k: int = 5,
) -> float:
    """Return the fraction of reference sources founds in the top-k"""
    if type(k) is not int or k < 0:
        raise ValueError("k must be a non-negative integer")

    if not reference_sources:
        raise ValueError("Recall requires at least one reference source")

    selected = retrieved_sources[:k]
    found_count = 0

    for reference in reference_sources:
        if source_is_found(reference, selected):
            found_count += 1

    return found_count / len(reference_sources)


def evaluate_recall(
    dataset: AnsweredDataset,
    results: StudentSearchResults,
    k: int = 5,
) -> tuple[dict[str, float], float]:
    """Return per-question recall and its ,ean over the dataset.

    Missing results count as zero. Duplicate or unknow question IDs
    and an empty reference dataset are rejected.
    """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a non-negative integer")

    if results.k < 0 or k > results.k:
        raise ValueError("Requested k exceeds the available result limit")

    if not dataset.rag_questions:
        raise ValueError("Evaluation requires a non empty dataset")

    reference_ids = {
        question.question_id
        for question in dataset.rag_questions
    }
    if len(reference_ids) != len(dataset.rag_questions):
        raise ValueError("Duplicate question_id in reference dataset")

    results_by_id = {
        result.question_id: result
        for result in results.search_results
    }
    if len(results_by_id) != len(results.search_results):
        raise ValueError("Duplicate question_id in search results")

    unknown_ids = set(results_by_id) - reference_ids
    if unknown_ids:
        raise ValueError(
            f"Unknow question_id in search results: {sorted(unknown_ids)}"
        )

    scores: dict[str, float] = {}

    for question in dataset.rag_questions:
        result = results_by_id.get(question.question_id)
        retrieved = [] if result is None else result.retrieved_sources

        scores[question.question_id] = recall_at_k(
            question.sources,
            retrieved,
            k=k,
        )

    mean_recall = sum(scores.values()) / len(scores)

    return scores, mean_recall
