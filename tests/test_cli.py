"""Tests for commands executed through the application entry point."""

import json

from pathlib import Path
import subprocess
import sys
import pytest

from src.indexer import load_index, save_index
from src.models import Chunk, ChunkIndex, StudentSearchResults


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_index_cli_rejects_invalid_size() -> None:
    """Reject invalid configuration without displaying a traceback."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "index",
            "--max_chunk_size",
            "0",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "max_chunk_size must be between 1 and 2000" in result.stderr
    assert "Traceback" not in result.stderr


def test_index_cli_accepts_configurable_paths(tmp_path: Path) -> None:
    """Build an index using explicitly supplied input and output paths."""
    processed_dir = tmp_path / "processed"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "index",
            "--raw_dir",
            "notes",
            "--processed_dir",
            str(processed_dir),
            "--max_chunk_size",
            "500",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    index = load_index(processed_dir)

    assert index.max_chunk_size == 500
    assert index.chunks
    assert all(
        chunk.file_path.startswith("notes/")
        for chunk in index.chunks
    )


@pytest.mark.parametrize("size", ["abc", "-1", "2001"])
def test_index_cli_rejects_invalid_size_arguments(size: str) -> None:
    """Reject malformed or out-of-range sizes without a traceback."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "index",
            "--max_chunk_size",
            size,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "max_chunk_size" in result.stderr
    assert "Traceback" not in result.stderr


def test_index_cli_reports_missing_corpus(tmp_path: Path) -> None:
    """Report a missing input directory without creating an index."""
    processed_dir = tmp_path / "processed"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "index",
            "--raw_dir",
            str(tmp_path / "missing"),
            "--processed_dir",
            str(processed_dir),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Indexing failed" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (processed_dir / "chunks.json").exists()


@pytest.fixture
def search_index_dir(tmp_path: Path) -> Path:
    """Save a small searchable index independently of the real corpus."""
    texts = [
        "load adapter",
        "install python",
        "configure server",
        "manage memory",
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

    processed_dir = tmp_path / "processed"
    save_index(index, processed_dir)

    return processed_dir


@pytest.mark.parametrize(
    ("query", "k", "expected_count"),
    [
        ("adapter", 5, 1),
        ("adapter", 0, 0),
        ("", 5, 0),
        ("xqzv", 5, 0),
    ],
)
def test_search_cli_writes_valid_results(
    search_index_dir: Path,
    tmp_path: Path,
    query: str,
    k: int,
    expected_count: int,
) -> None:
    """Save the expected results through the real CLI entry point."""
    output_path = tmp_path / "output" / "result.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "search",
            query,
            "--k",
            str(k),
            "--processed_dir",
            str(search_index_dir),
            "--output_path",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    output = StudentSearchResults.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )

    assert output.k == k
    assert len(output.search_results) == 1

    answer = output.search_results[0]

    assert answer.question == query
    assert answer.question_id
    assert len(answer.retrieved_sources) == expected_count

    if expected_count:
        source = answer.retrieved_sources[0]
        assert source.file_path == "data/raw/example_0.txt"
        assert source.first_character_index == 0
        assert source.last_character_index == len("load adapter")


@pytest.mark.parametrize("k", ["-1", "abc"])
def test_search_cli_rejects_invalid_k(
    tmp_path: Path,
    k: str,
) -> None:
    """Reject invalid k before attempting to load an index."""
    output_path = tmp_path / "result.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "search",
            "adapter",
            "--k",
            k,
            "--processed_dir",
            str(tmp_path / "missing"),
            "--output_path",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "k must be a non-negative integer" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_path.exists()


def test_search_cli_reports_missing_index(tmp_path: Path) -> None:
    """Report an absent index without creating search results."""
    output_path = tmp_path / "result.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "search",
            "adapter",
            "--processed_dir",
            str(tmp_path / "missing"),
            "--output_path",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Search failed" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_path.exists()


def test_search_dataset_cli_writes_valid_results(
    search_index_dir: Path,
    tmp_path: Path,
) -> None:
    """Search a dataset using configurable input and output paths."""
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        '{"rag_questions": ['
        '{"question_id": "q42", "question": "adapter"},'
        '{"question_id": "q43", "question": "xqzv"}'
        ']}',
        encoding="utf-8",
    )
    save_directory = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "search_dataset",
            "--dataset_path",
            str(dataset_path),
            "--k",
            "5",
            "--processed_dir",
            str(search_index_dir),
            "--save_directory",
            str(save_directory),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    target = save_directory / dataset_path.name
    output = StudentSearchResults.model_validate_json(
        target.read_text(encoding="utf-8")
    )

    assert output.k == 5
    assert [
        question.question_id
        for question in output.search_results
    ] == ["q42", "q43"]
    assert [
        question.question
        for question in output.search_results
    ] == ["adapter", "xqzv"]

    sources = output.search_results[0].retrieved_sources
    assert len(sources) == 1
    assert sources[0].file_path == "data/raw/example_0.txt"
    assert sources[0].first_character_index == 0
    assert sources[0].last_character_index == len("load adapter")

    assert output.search_results[1].retrieved_sources == []


@pytest.mark.parametrize("k", ["-1", "abc"])
def test_search_dataset_cli_rejects_invalid_k(
    tmp_path: Path,
    k: str,
) -> None:
    """Reject invalid k before loading any input file."""
    save_directory = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "search_dataset",
            "--dataset_path",
            str(tmp_path / "missing.json"),
            "--k",
            k,
            "--processed_dir",
            str(tmp_path / "missing_index"),
            "--save_directory",
            str(save_directory),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "k must be a non-negative integer" in result.stderr
    assert "Traceback" not in result.stderr
    assert not save_directory.exists()


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("missing_dataset", "Dataset search failed"),
        ("invalid_json", "Invalid JSON"),
        ("missing_index", "Dataset search failed"),
    ],
)
def test_search_dataset_cli_reports_invalid_inputs(
    tmp_path: Path,
    case: str,
    expected_message: str,
) -> None:
    """Report missing files or malformed JSON without a traceback."""
    dataset_path = tmp_path / "questions.json"
    save_directory = tmp_path / "output"

    if case == "invalid_json":
        dataset_path.write_text("{", encoding="utf-8")
    elif case == "missing_index":
        dataset_path.write_text(
            '{"rag_questions": ['
            '{"question_id": "q42", "question": "adapter"}'
            ']}',
            encoding="utf-8",
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "search_dataset",
            "--dataset_path",
            str(dataset_path),
            "--processed_dir",
            str(tmp_path / "missing_index"),
            "--save_directory",
            str(save_directory),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert expected_message in result.stderr
    assert "Traceback" not in result.stderr
    assert not save_directory.exists()


@pytest.mark.parametrize(
    ("dataset_json", "expected_ids"),
    [
        (
            '{"rag_questions": ['
            '{"question_id": "q42", "question": "adapter"}'
            ']}',
            ["q42"],
        ),
        ('{"rag_questions": []}', []),
    ],
)
def test_search_dataset_cli_handles_zero_k_and_empty_dataset(
    search_index_dir: Path,
    tmp_path: Path,
    dataset_json: str,
    expected_ids: list[str],
) -> None:
    """Preserve questions with k=0 and accept an empty dataset."""
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(dataset_json, encoding="utf-8")
    save_directory = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "search_dataset",
            "--dataset_path",
            str(dataset_path),
            "--k",
            "0",
            "--processed_dir",
            str(search_index_dir),
            "--save_directory",
            str(save_directory),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

    target = save_directory / dataset_path.name
    output = StudentSearchResults.model_validate_json(
        target.read_text(encoding="utf-8")
    )

    assert output.k == 0
    assert [
        question.question_id
        for question in output.search_results
    ] == expected_ids
    assert all(
        question.retrieved_sources == []
        for question in output.search_results
    )


@pytest.mark.parametrize(
    ("k", "expected_score"),
    [
        (5, "100.00%"),
        (0, "0.00%"),
    ],
)
def test_evaluate_cli_reports_recall(
    tmp_path: Path,
    k: int,
    expected_score: str,
) -> None:
    """Evaluate JSON files through the real CLI entry point."""
    source = {
        "file_path": "data/raw/example.py",
        "first_character_index": 100,
        "last_character_index": 200,
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps({
            "rag_questions": [{
                "question_id": "q1",
                "question": "Example question",
                "sources": [source],
                "answer": "Reference answer",
            }],
        }),
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps({
            "search_results": [{
                "question_id": "q1",
                "question": "Example question",
                "retrieved_sources": [source],
            }],
            "k": 5,
        }),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "evaluate",
            "--student_search_results_path",
            str(results_path),
            "--dataset_path",
            str(reference_path),
            "--k",
            str(k),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Evaluated 1 questions" in result.stdout
    assert "Questions missing from results: 0" in result.stdout
    assert f"Recall@{k}: {expected_score}" in result.stdout


@pytest.mark.parametrize(
    ("case", "expected_message"),
    [
        ("missing_reference", "Evaluation failed"),
        ("missing_results", "Evaluation failed"),
        ("invalid_reference", "Evaluation failed"),
        ("invalid_results", "Evaluation failed"),
        ("negative_k", "k must be a non-negative integer"),
        ("text_k", "k must be a non-negative integer"),
    ],
)
def test_evaluate_cli_reports_input_errors(
    tmp_path: Path,
    case: str,
    expected_message: str,
) -> None:
    """Report invalid inputs without an unhandled traceback."""
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        '{"rag_questions": []}',
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(
        '{"search_results": [], "k": 5}',
        encoding="utf-8",
    )
    k = "5"

    if case == "missing_reference":
        reference_path = tmp_path / "missing_reference.json"
    elif case == "missing_results":
        results_path = tmp_path / "missing_results.json"
    elif case == "invalid_reference":
        reference_path.write_text("{", encoding="utf-8")
    elif case == "invalid_results":
        results_path.write_text("{", encoding="utf-8")
    elif case == "negative_k":
        k = "-1"
    elif case == "text_k":
        k = "abc"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "evaluate",
            "--student_search_results_path",
            str(results_path),
            "--dataset_path",
            str(reference_path),
            "--k",
            k,
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert expected_message in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("arguments", "expected_message"),
    [
        (
            ["", "--k", "3"],
            "query must not be empty",
        ),
        (
            ["adapter", "--k", "-1"],
            "k must be a non-negative integer",
        ),
        (
            ["adapter", "--k", "abc"],
            "k must be a non-negative integer",
        ),
        (
            ["adapter", "--context_budget", "0"],
            "context_budget must be a positive integer",
        ),
        (
            ["adapter", "--max_new_tokens", "0"],
            "max_new_tokens must be positive",
        ),
        (
            [
                "adapter",
                "--context_budget",
                "128",
                "--max_new_tokens",
                "128",
            ],
            "max_new_tokens must be positive",
        ),
    ],
)
def test_answer_cli_rejects_invalid_arguments(
    tmp_path: Path,
    arguments: list[str],
    expected_message: str,
) -> None:
    """Reject invalid arguments through Fire without a traceback."""
    target = tmp_path / "answer.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "answer",
            *arguments,
            "--processed_dir",
            str(tmp_path / "missing"),
            "--output_path",
            str(target),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert expected_message in result.stderr
    assert "Traceback" not in result.stderr
    assert not target.exists()


@pytest.mark.parametrize(
    "input_kind",
    ["missing", "invalid_json", "invalid_schema"],
)
def test_answer_dataset_cli_reports_invalid_input(
    tmp_path: Path,
    input_kind: str,
) -> None:
    """Reject invalid batch input without a traceback or output."""
    source = tmp_path / "questions.json"
    output_directory = tmp_path / "answers"

    if input_kind == "invalid_json":
        source.write_text("{", encoding="utf-8")
    elif input_kind == "invalid_schema":
        source.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "answer_dataset",
            "--student_search_results_path",
            str(source),
            "--save_directory",
            str(output_directory),
            "--project_root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert "Dataset answer generation failed:" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Loading Qwen..." not in result.stdout
    assert not output_directory.exists()


def test_answer_dataset_cli_handles_empty_dataset(tmp_path: Path) -> None:
    """Save valid empty output without loading the language model."""
    source = tmp_path / "empty.json"
    output_directory = tmp_path / "answers"

    source.write_text(
        json.dumps({"search_results": [], "k": 5}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src",
            "answer_dataset",
            "--student_search_results_path",
            str(source),
            "--save_directory",
            str(output_directory),
            "--project_root",
            str(tmp_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr
    assert "Loading Qwen..." not in result.stdout
    assert "Loaded 0 questions" in result.stdout
    assert "Processed 0 of 0 questions" in result.stdout
    assert "Saved student_search_results_and_answer to" in result.stdout

    target = output_directory / source.name
    saved = json.loads(target.read_text(encoding="utf-8"))

    assert saved == {"search_results": [], "k": 5}
