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


def test_load_source_chunk_preserves_text_and_offsets(
    tmp_path: Path,
) -> None:
    """Read Unicode text with the same newline convention as ingestion."""
    from src.datasets import load_source_chunk

    target = tmp_path / "example.md"
    target.write_bytes("é\r\nALPHA\r\n".encode("utf-8"))

    source = MinimalSource(
        file_path="example.md",
        first_character_index=2,
        last_character_index=7,
    )

    chunk = load_source_chunk(source, tmp_path)

    assert chunk.text == "ALPHA"
    assert chunk.file_path == source.file_path
    assert chunk.first_character_index == 2
    assert chunk.last_character_index == 7


def test_load_source_chunk_rejects_out_of_bounds(
    tmp_path: Path,
) -> None:
    """Reject an end position beyond the actual file content."""
    from src.datasets import load_source_chunk

    (tmp_path / "example.md").write_text("short", encoding="utf-8")

    source = MinimalSource(
        file_path="example.md",
        first_character_index=0,
        last_character_index=20,
    )

    with pytest.raises(ValueError, match="exceeds file length"):
        load_source_chunk(source, tmp_path)


def test_answer_questions_preserves_ids_and_reuses_generator(
    tmp_path: Path,
) -> None:
    """Answer two questions with the same generator and preserve IDs."""
    from unittest.mock import Mock

    from src.datasets import answer_questions

    target = tmp_path / "example.md"
    target.write_text("ALPHA passage", encoding="utf-8")

    source = MinimalSource(
        file_path="example.md",
        first_character_index=0,
        last_character_index=13,
    )

    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id=question_id,
                question=question,
                retrieved_sources=[source],
            )
            for question_id, question in [
                ("q1", "What is ALPHA?"),
                ("q2", "Describe this passage."),
            ]
        ],
        k=1,
    )

    generator = Mock()
    generator.context_budget = 4096
    generator.max_new_tokens = 256
    generator.tokenizer.apply_chat_template.return_value = [0] * 20
    generator.generate.side_effect = [
        "First controlled answer.",
        "Second controlled answer.",
    ]

    output = answer_questions(results, generator, tmp_path)

    assert output.k == 1
    assert [
        answer.question_id
        for answer in output.search_results
    ] == ["q1", "q2"]

    assert [
        answer.question
        for answer in output.search_results
    ] == [
        "What is ALPHA?",
        "Describe this passage.",
    ]

    assert [
        answer.answer
        for answer in output.search_results
    ] == [
        "First controlled answer.",
        "Second controlled answer.",
    ]

    assert generator.generate.call_count == 2

    for answer in output.search_results:
        assert answer.retrieved_sources == [source]

    for call in generator.generate.call_args_list:
        prompt = call.args[0]
        assert "ALPHA passage" in prompt.messages[1].content
        assert prompt.sources == [source]