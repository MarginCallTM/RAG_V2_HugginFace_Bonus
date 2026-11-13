"""Tests for corpus discovery and reading."""

from pathlib import Path

import pytest

from src.ingest import list_corpus_files, read_text, to_corpus_path


def test_list_corpus_files_filters_and_sorts(tmp_path: Path) -> None:
    """Select supported files recursively in sorted order."""
    nested = tmp_path / "nested"
    nested.mkdir()

    expected = [
        tmp_path / "a.md",
        nested / "b.py",
        nested / "c.txt",
    ]

    for path in expected:
        path.write_text("example", encoding="utf-8")

    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")
    (tmp_path / "directory.py").mkdir()

    files = list_corpus_files(tmp_path)

    assert files == sorted(expected)


def test_list_corpus_files_rejects_missing_directory(
    tmp_path: Path,
) -> None:
    """Report a missing corpus instead of returning an empty list."""
    with pytest.raises(NotADirectoryError):
        list_corpus_files(tmp_path / "missing")


def test_read_text_decodes_utf8(tmp_path: Path) -> None:
    """Preserve non-ASCII characters when decoding UTF-8."""
    path = tmp_path / "example.txt"
    path.write_bytes("Modèle français : été.".encode("utf-8"))

    assert read_text(path) == "Modèle français : été."


def test_read_text_normalizes_crlf(tmp_path: Path) -> None:
    """Apply the chosen newline convention consistently."""
    path = tmp_path / "example.txt"
    path.write_bytes(b"ABC\r\nDEF")

    assert read_text(path) == "ABC\nDEF"


def test_read_text_rejects_invalid_utf8(tmp_path: Path) -> None:
    """Report invalid encoding instead of altering the content."""
    path = tmp_path / "example.txt"
    path.write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        read_text(path)


def test_to_corpus_path_returns_project_relative_path(
    tmp_path: Path,
) -> None:
    """Remove the machine-specific project prefix."""
    source = tmp_path / "data" / "raw" / "example.py"

    result = to_corpus_path(source, tmp_path)

    assert result == "data/raw/example.py"


def test_to_corpus_path_rejects_external_path(
    tmp_path: Path,
) -> None:
    """Reject sources outside the chosen project root."""
    project_root = tmp_path / "project"
    external_source = tmp_path / "outside.py"

    with pytest.raises(ValueError):
        to_corpus_path(external_source, project_root)
