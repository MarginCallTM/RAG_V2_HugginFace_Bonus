"""Tests for the RAG data contracts."""

import pytest
from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    UnansweredQuestion,
    RagDataset,
    AnsweredDataset,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
    Chunk,
    ChunkIndex,
)


def test_source_accepts_one_character() -> None:
    """Accept the smallest non-empty character range."""
    source = MinimalSource(
        file_path="example.py",
        first_character_index=5,
        last_character_index=6,
    )

    assert source.last_character_index - source.first_character_index == 1


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, 10),
        (5, 5),
        (10, 5),
    ],
)
def test_source_rejects_invalid_range(start: int, end: int) -> None:
    """Reject negative starts, empty ranges and reversed ranges."""
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="example.py",
            first_character_index=start,
            last_character_index=end,
        )


def test_question_preserves_provided_id() -> None:
    """Keep the identifier supplied by the dataset."""
    question = UnansweredQuestion(
        question_id="question-42",
        question="How does LoRA work?",
    )

    assert question.question_id == "question-42"


def test_questions_generate_distinct_ids() -> None:
    """Generate a fresh identifier when none is supplied."""
    first = UnansweredQuestion(question="First question")
    second = UnansweredQuestion(question="Second question")

    assert first.question_id != second.question_id


def test_answered_question_rejects_invalid_source() -> None:
    """Validate sources nested inside an answered question."""
    data = {
        "question_id": "question-42",
        "question": "How does LoRA work?",
        "answer": "Reference answer.",
        "sources": [
            {
                "file_path": "example.py",
                "first_character_index": 10,
                "last_character_index": 5,
            }
        ],
    }

    with pytest.raises(ValidationError):
        AnsweredQuestion.model_validate(data)


def test_rag_dataset_falls_back_to_unanswered_question() -> None:
    """Show that the union can discard invalid reference data."""
    data = {
        "rag_questions": [
            {
                "question_id": "q1",
                "question": "How does LoRA work?",
                "answer": "Reference answer.",
                "sources": [
                    {
                        "file_path": "example.py",
                        "first_character_index": 10,
                        "last_character_index": 5,
                    }
                ],
            }
        ]
    }

    dataset = RagDataset.model_validate(data)
    question = dataset.rag_questions[0]

    assert isinstance(question, UnansweredQuestion)
    assert not isinstance(question, AnsweredQuestion)
    assert "sources" not in question.model_dump()
    assert "answer" not in question.model_dump()


def test_answered_dataset_rejects_invalid_reference() -> None:
    """Reject invalid sources instead of losing reference data."""
    data = {
        "rag_questions": [
            {
                "question_id": "q1",
                "question": "How does LoRA work?",
                "answer": "Reference answer.",
                "sources": [
                    {
                        "file_path": "example.py",
                        "first_character_index": 10,
                        "last_character_index": 5,
                    }
                ],
            }
        ]
    }

    with pytest.raises(ValidationError):
        AnsweredDataset.model_validate(data)


def test_search_results_json_round_trip() -> None:
    """Preserve retrieval results through JSON serialization."""
    source = MinimalSource(
        file_path="example.py",
        first_character_index=0,
        last_character_index=100,
    )

    result = MinimalSearchResults(
        question_id="q1",
        question="How does LoRA work?",
        retrieved_sources=[source],
    )

    batch = StudentSearchResults(
        search_results=[result],
        k=5,
    )

    serialized = batch.model_dump_json()
    restored = StudentSearchResults.model_validate_json(serialized)

    assert restored == batch
    assert restored.k == 5
    assert len(restored.search_results[0].retrieved_sources) == 1


def test_generated_answer_json_round_trip() -> None:
    """Preserve an answer and its sources through JSON serialization."""
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
        first_character_index=4695,
        last_character_index=6098,
    )

    answer = MinimalAnswer(
        question_id="q1",
        question="Which endpoint loads a LoRA adapter?",
        retrieved_sources=[source],
        answer="Send a POST request to /v1/load_lora_adapter.",
    )

    batch = StudentSearchResultsAndAnswer(
        search_results=[answer],
        k=5,
    )

    serialized = batch.model_dump_json()
    restored = StudentSearchResultsAndAnswer.model_validate_json(
        serialized
    )

    assert restored == batch
    assert restored.search_results[0].retrieved_sources == [source]
    assert restored.search_results[0].answer == answer.answer


def test_chunk_index_json_round_trip() -> None:
    """Preserve index configuration and nested chunks through JSON."""
    chunk = Chunk(
        file_path="example.txt",
        first_character_index=0,
        last_character_index=7,
        text="Bonjour",
    )
    index = ChunkIndex(
        max_chunk_size=10,
        chunks=[chunk],
    )

    serialized = index.model_dump_json()
    restored = ChunkIndex.model_validate_json(serialized)

    assert restored == index
    assert isinstance(restored.chunks[0], Chunk)


def test_chunk_index_rejects_chunk_above_configured_limit() -> None:
    """Reject a chunk exceeding the index's declared size limit."""
    chunk = Chunk(
        file_path="example.txt",
        first_character_index=0,
        last_character_index=7,
        text="Bonjour",
    )

    with pytest.raises(ValidationError):
        ChunkIndex(
            max_chunk_size=5,
            chunks=[chunk],
        )


@pytest.mark.parametrize("size", [0, -1, 2001])
def test_chunk_index_rejects_invalid_size(size: int) -> None:
    """Reject configuration values outside the project's size limits."""
    with pytest.raises(ValidationError):
        ChunkIndex(
            max_chunk_size=size,
            chunks=[],
        )
