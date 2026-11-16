
"""Test answer orchestration without loading model weights."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.cli import CliError, RagCli
from src.indexer import save_index
from src.models import (
    Chunk,
    ChunkIndex,
    StudentSearchResultsAndAnswer,
)


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """Save a small corpus with two passages matching adapter."""
    texts = [
        "adapter alpha",
        "adapter beta",
        "python installation",
        "server configuration",
        "memory management",
    ]

    index = ChunkIndex(
        max_chunk_size=2000,
        chunks=[
            Chunk(
                file_path=f"data/raw/example_{number}.txt",
                first_character_index=0,
                last_character_index=len(text),
                text=text,
            )
            for number, text in enumerate(texts)
        ],
    )

    directory = tmp_path / "processed"
    save_index(index, directory)
    return directory


@pytest.fixture
def fake_generator() -> Mock:
    """Provide a generator with controlled answer and token counts."""
    generator = Mock()
    generator.context_budget = 400
    generator.max_new_tokens = 256
    generator.generate.return_value = "A controlled answer."

    def tokenize(
        messages: list[dict[str, str]],
        **kwargs: object,
    ) -> list[int]:
        """Simulate ten base tokens and one hundred per passage."""
        passage_count = messages[1]["content"].count("[Source ")
        return [0] * (10 + 100 * passage_count)

    generator.tokenizer.apply_chat_template.side_effect = tokenize
    return generator


def test_answer_saves_only_included_sources(
    processed_dir: Path,
    tmp_path: Path,
    fake_generator: Mock,
) -> None:
    """Save the generated text and omit the passage above the budget."""
    target = tmp_path / "answers" / "result.json"

    with patch(
        "src.generator.Generator",
        return_value=fake_generator,
    ) as constructor:
        RagCli().answer(
            "adapter",
            k=3,
            processed_dir=str(processed_dir),
            output_path=str(target),
            context_budget=400,
            max_new_tokens=256,
        )

    constructor.assert_called_once_with(
        model_name="Qwen/Qwen3-0.6B",
        context_budget=400,
        max_new_tokens=256,
    )
    fake_generator.generate.assert_called_once()

    prompt = fake_generator.generate.call_args.args[0]
    output = StudentSearchResultsAndAnswer.model_validate_json(
        target.read_text(encoding="utf-8")
    )

    assert output.k == 3
    assert len(output.search_results) == 1

    result = output.search_results[0]

    assert result.question_id
    assert result.question == "adapter"
    assert result.answer == "A controlled answer."
    assert len(result.retrieved_sources) == 1
    assert result.retrieved_sources == prompt.sources
    assert result.retrieved_sources[0].file_path == (
        "data/raw/example_0.txt"
    )
    assert not target.with_name(target.name + ".tmp").exists()


@pytest.mark.parametrize(
    ("query", "k"),
    [
        ("adapter", 0),
        ("xqzv", 3),
    ],
)
def test_answer_without_matches_does_not_load_model(
    processed_dir: Path,
    tmp_path: Path,
    query: str,
    k: int,
) -> None:
    """Save an insufficient-evidence answer without loading Qwen."""
    target = tmp_path / "answer.json"

    with patch("src.generator.Generator") as constructor:
        RagCli().answer(
            query,
            k=k,
            processed_dir=str(processed_dir),
            output_path=str(target),
        )

    constructor.assert_not_called()

    output = StudentSearchResultsAndAnswer.model_validate_json(
        target.read_text(encoding="utf-8")
    )
    result = output.search_results[0]

    assert result.retrieved_sources == []
    assert "Insufficient evidence" in result.answer


def test_answer_reports_generation_failure(
    processed_dir: Path,
    tmp_path: Path,
    fake_generator: Mock,
) -> None:
    """Turn an expected model failure into a CLI error."""
    target = tmp_path / "answer.json"
    fake_generator.generate.side_effect = RuntimeError(
        "Simulated model failure"
    )

    with patch(
        "src.generator.Generator",
        return_value=fake_generator,
    ):
        with pytest.raises(
            CliError,
            match="Answer generation failed: Simulated model failure",
        ):
            RagCli().answer(
                "adapter",
                processed_dir=str(processed_dir),
                output_path=str(target),
                context_budget=400,
                max_new_tokens=256,
            )

    assert not target.exists()


def test_answer_reports_missing_index(tmp_path: Path) -> None:
    """Reject an absent index without loading the model."""
    with patch("src.generator.Generator") as constructor:
        with pytest.raises(CliError, match="Answer generation failed"):
            RagCli().answer(
                "adapter",
                processed_dir=str(tmp_path / "missing"),
                output_path=str(tmp_path / "answer.json"),
            )

    constructor.assert_not_called()
