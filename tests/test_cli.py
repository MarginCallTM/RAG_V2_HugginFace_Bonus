"""Tests for commands executed through the application entry point."""

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
