"""Test prompt provenance and budget selection without downloading Qwen."""

from typing import cast
from unittest.mock import Mock

import pytest
from transformers import PreTrainedTokenizerBase

from src.models import Chunk
from src.prompting import (
    build_budgeted_prompt,
    build_prompt,
    count_prompt_tokens,
)


@pytest.fixture
def chunks() -> list[Chunk]:
    """Provide two passages with distinct texts and source positions."""
    return [
        Chunk(
            file_path="data/raw/first.md",
            first_character_index=10,
            last_character_index=15,
            text="ALPHA",
        ),
        Chunk(
            file_path="data/raw/second.py",
            first_character_index=20,
            last_character_index=24,
            text="BETA",
        ),
    ]


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    """Simulate token counts and check chat-template arguments."""
    mock = Mock(spec=PreTrainedTokenizerBase)

    def apply_template(
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        """Use message characters as deterministic fake tokens."""
        assert tokenize is True
        assert add_generation_prompt is True
        assert enable_thinking is False

        size = sum(len(message["content"]) for message in messages)
        return [0] * size

    mock.apply_chat_template.side_effect = apply_template
    return cast(PreTrainedTokenizerBase, mock)


def test_build_prompt_preserves_text_and_provenance(
    chunks: list[Chunk],
) -> None:
    """Keep passage order, exact text and matching source positions."""
    prompt = build_prompt("Which passage?", chunks)

    assert [message.role for message in prompt.messages] == [
        "system",
        "user",
    ]

    content = prompt.messages[1].content

    for number, chunk in enumerate(chunks, start=1):
        expected = (
            f"[Source {number}] {chunk.file_path}\n"
            f"Characters [{chunk.first_character_index}:"
            f"{chunk.last_character_index}]\n"
            f"{chunk.text}"
        )
        assert expected in content

    assert content.index("[Source 1]") < content.index("[Source 2]")
    assert content.endswith("Question:\nWhich passage?")
    assert [source.model_dump() for source in prompt.sources] == [
        chunk.model_dump(exclude={"text"})
        for chunk in chunks
    ]


def test_build_prompt_accepts_no_evidence() -> None:
    """Keep the question usable when retrieval returns no passages."""
    prompt = build_prompt("Unknown topic?", [])

    assert prompt.sources == []
    assert "No evidence retrieved" in prompt.messages[1].content
    assert "Unknown topic?" in prompt.messages[1].content


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_build_prompt_rejects_blank_question(question: str) -> None:
    """Reject questions containing no meaningful characters."""
    with pytest.raises(ValueError, match="Question must not be empty"):
        build_prompt(question, [])


def test_count_prompt_tokens_uses_complete_messages(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Count both system and user content with the fake tokenizer."""
    prompt = build_prompt("Example?", [])
    expected = sum(len(message.content) for message in prompt.messages)

    assert count_prompt_tokens(prompt, tokenizer) == expected


def test_budget_keeps_all_passages_when_they_fit(
    chunks: list[Chunk],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Accept a prompt exactly at the input budget boundary."""
    full = build_prompt("Example?", chunks)
    size = count_prompt_tokens(full, tokenizer)

    result = build_budgeted_prompt(
        "Example?",
        chunks,
        tokenizer,
        context_budget=size + 32,
        max_new_tokens=32,
    )

    assert result == full


def test_budget_excludes_passage_that_does_not_fit(
    chunks: list[Chunk],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Keep the first passage and omit the second text and source."""
    first = build_prompt("Example?", chunks[:1])
    size = count_prompt_tokens(first, tokenizer)

    result = build_budgeted_prompt(
        "Example?",
        chunks,
        tokenizer,
        context_budget=size + 32,
        max_new_tokens=32,
    )

    assert result == first
    assert "BETA" not in result.messages[1].content


def test_budget_can_return_no_passages(
    chunks: list[Chunk],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Keep a valid empty-context prompt if the first chunk cannot fit."""
    empty = build_prompt("Example?", [])
    size = count_prompt_tokens(empty, tokenizer)

    result = build_budgeted_prompt(
        "Example?",
        chunks,
        tokenizer,
        context_budget=size + 32,
        max_new_tokens=32,
    )

    assert result == empty


def test_budget_rejects_question_that_does_not_fit(
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Reject oversized instructions and question before adding chunks."""
    with pytest.raises(ValueError, match="exceed the input token budget"):
        build_budgeted_prompt(
            "Long question " * 100,
            [],
            tokenizer,
            context_budget=128,
            max_new_tokens=32,
        )


@pytest.mark.parametrize(
    ("context_budget", "max_new_tokens"),
    [(0, 32),
        (-1, 32),
        (128, 0),
        (128, -1),
        (128, 128),
        (128, 129),
     ],
)
def test_budget_rejects_invalid_limits(
    tokenizer: PreTrainedTokenizerBase,
    context_budget: int,
    max_new_tokens: int,
) -> None:
    """Reject invalid total budgets and output reservations."""
    with pytest.raises(ValueError):
        build_budgeted_prompt(
            "Example?",
            [],
            tokenizer,
            context_budget=context_budget,
            max_new_tokens=max_new_tokens,
        )
