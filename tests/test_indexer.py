"""Tests for chunk-index construction and persistence."""

from pathlib import Path

import pytest

from pydantic import ValidationError

from src.indexer import build_index, load_index, save_index
from src.models import Chunk, ChunkIndex


def test_build_index_collects_chunks_with_source_paths(
    tmp_path: Path,
) -> None:
    """Build a validated index with project-relative source paths."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    source = raw_dir / "example.txt"
    source.write_text("ABCDEF", encoding="utf-8")

    index = build_index(
        raw_dir=raw_dir,
        project_root=tmp_path,
        max_chunk_size=3,
    )

    assert index.max_chunk_size == 3
    assert [chunk.text for chunk in index.chunks] == ["ABC", "DEF"]
    assert [
        (chunk.first_character_index, chunk.last_character_index)
        for chunk in index.chunks
    ] == [(0, 3), (3, 6)]
    assert all(
        chunk.file_path == "data/raw/example.txt"
        for chunk in index.chunks
    )


def test_build_index_rejects_empty_corpus(tmp_path: Path) -> None:
    """Refuse to return an index without chunks."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match="no indexable text"):
        build_index(raw_dir, tmp_path)


def test_build_index_rejects_corpus_with_only_empty_files(
    tmp_path: Path,
) -> None:
    """Distinguish file discovery from usable text content."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "empty.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no indexable text"):
        build_index(raw_dir, tmp_path)


def test_save_and_load_index_preserve_data(tmp_path: Path) -> None:
    """Preserve configuration and chunks through a disk round trip."""
    index = ChunkIndex(
        max_chunk_size=10,
        chunks=[
            Chunk(
                file_path="example.txt",
                first_character_index=0,
                last_character_index=7,
                text="Bonjour",
            )
        ],
    )
    processed_dir = tmp_path / "processed"

    target = save_index(index, processed_dir)
    restored = load_index(processed_dir)

    assert target == processed_dir / "chunks.json"
    assert target.is_file()
    assert not (processed_dir / "chunks.json.tmp").exists()
    assert restored == index
    assert isinstance(restored.chunks[0], Chunk)


def test_load_index_reports_missing_file(tmp_path: Path) -> None:
    """Report an index that has not been saved."""
    with pytest.raises(FileNotFoundError):
        load_index(tmp_path)


def test_load_index_rejects_inconsistent_data(tmp_path: Path) -> None:
    """Validate saved data instead of trusting the JSON file."""
    target = tmp_path / "chunks.json"
    target.write_text(
        '{"max_chunk_size": 0, "chunks": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_index(tmp_path)
