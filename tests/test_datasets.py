"""Tests for dataset loading and retrieval-result persistence."""

from pathlib import Path

import pytest

from src.datasets import load_dataset, save_search_results
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def test_load_dataset_preserves_question_id(tmp_path: Path) -> None:
    """Keep the identifier and question supplied in the JSON file."""
    target = tmp_path / "dataset.json"
    target.write_text(
        '{"rag_questions": ['
        '{"question_id": "q42", "question": "How does LoRA work?"}'
        ']}',
        encoding="utf-8",
    )

    dataset = load_dataset(target)

    assert len(dataset.rag_questions) == 1
    assert dataset.rag_questions[0].question_id == "q42"
    assert dataset.rag_questions[0].question == "How does LoRA work?"


def test_save_search_results_round_trip(tmp_path: Path) -> None:
    """Preserve results through a save and reload on disk."""
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q42",
                question="How does LoRA work?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/example.md",
                        first_character_index=10,
                        last_character_index=2010,
                    )
                ],
            )
        ],
        k=5,
    )
    target = tmp_path / "output" / "results.json"

    saved_path = save_search_results(results, target)
    restored = StudentSearchResults.model_validate_json(
        target.read_text(encoding="utf-8")
    )

    assert saved_path == target
    assert restored == results
    assert not target.with_name(target.name + ".tmp").exists()


def test_save_search_results_rejects_more_than_k_sources(
    tmp_path: Path,
) -> None:
    """Reject a question containing more sources than requested."""
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q42",
                question="How does LoRA work?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/example.md",
                        first_character_index=0,
                        last_character_index=100,
                    )
                ],
            )
        ],
        k=0,
    )
    target = tmp_path / "results.json"

    with pytest.raises(ValueError, match="more than k sources"):
        save_search_results(results, target)

    assert not target.exists()


def test_save_search_results_rejects_oversized_source(
    tmp_path: Path,
) -> None:
    """Reject a source whose range exceeds 2000 characters."""
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q42",
                question="How does LoRA work?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/example.md",
                        first_character_index=10,
                        last_character_index=2011,
                    )
                ],
            )
        ],
        k=1,
    )
    target = tmp_path / "results.json"

    with pytest.raises(ValueError, match="exceeds 2000 characters"):
        save_search_results(results, target)

    assert not target.exists()
