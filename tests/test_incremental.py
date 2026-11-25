"""Verify incremental chunk results and the work avoided."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.chunking import chunk_file
from src.incremental import rebuild_file_chunks, update_chunk_index
from src.indexer import build_index
from src.manifest import build_manifest


def test_update_rebuilds_only_added_and_modified_files(
    tmp_path: Path,
) -> None:
    """Match a full build without chunking unchanged or deleted files."""
    for name in ("keep.txt", "edit.txt", "remove.txt"):
        (tmp_path / name).write_text("port 8000\n", encoding="utf-8")
    previous_manifest = build_manifest(tmp_path, tmp_path, 10)
    previous_index = build_index(tmp_path, tmp_path, 10)
    original_data = previous_index.model_dump()

    (tmp_path / "edit.txt").write_text(
        "New introduction.\nport 9000\n", encoding="utf-8",
    )
    (tmp_path / "remove.txt").unlink()
    (tmp_path / "add.md").write_text("# Guide\nHello\n", encoding="utf-8")
    current_manifest = build_manifest(tmp_path, tmp_path, 10)

    with patch("src.incremental.chunk_file", wraps=chunk_file) as observed:
        updated = update_chunk_index(
            previous_index, previous_manifest, current_manifest, tmp_path,
        )

    assert [call.args[0] for call in observed.call_args_list] == [
        "add.md", "edit.txt",
    ]
    assert updated == build_index(tmp_path, tmp_path, 10)
    assert previous_index.model_dump() == original_data


def test_update_without_changes_does_not_read_or_chunk(
    tmp_path: Path,
) -> None:
    """Reuse existing chunks after scanning an unchanged corpus."""
    (tmp_path / "keep.txt").write_text("Keep this text.\n", encoding="utf-8")
    previous_manifest = build_manifest(tmp_path, tmp_path, 10)
    previous_index = build_index(tmp_path, tmp_path, 10)
    current_manifest = build_manifest(tmp_path, tmp_path, 10)

    with (
        patch("src.incremental.chunk_file", wraps=chunk_file) as observed,
        patch("src.incremental.read_text") as read,
    ):
        updated = update_chunk_index(
            previous_index, previous_manifest, current_manifest, tmp_path,
        )

    observed.assert_not_called()
    read.assert_not_called()
    assert updated == previous_index


def test_update_changed_configuration_rebuilds_all_files(
    tmp_path: Path,
) -> None:
    """Do not reuse old chunks after changing their size budget."""
    for name in ("b.txt", "a.txt"):
        (tmp_path / name).write_text(
            "Long enough for chunks.\n", encoding="utf-8",
        )
    previous_manifest = build_manifest(tmp_path, tmp_path, 20)
    previous_index = build_index(tmp_path, tmp_path, 20)
    current_manifest = build_manifest(tmp_path, tmp_path, 10)

    with patch("src.incremental.chunk_file", wraps=chunk_file) as observed:
        updated = update_chunk_index(
            previous_index, previous_manifest, current_manifest, tmp_path,
        )

    assert [call.args[0] for call in observed.call_args_list] == [
        "a.txt", "b.txt",
    ]
    assert updated == build_index(tmp_path, tmp_path, 10)


def test_rebuild_rejects_change_after_inventory(tmp_path: Path) -> None:
    """Reject stale fingerprints before running the chunker."""
    path = tmp_path / "server.txt"
    path.write_text("port 8000", encoding="utf-8")
    manifest = build_manifest(tmp_path, tmp_path)
    path.write_text("port 9000", encoding="utf-8")

    with patch("src.incremental.chunk_file", wraps=chunk_file) as observed:
        with pytest.raises(ValueError, match="File changed after inventory"):
            rebuild_file_chunks(manifest.files[0], tmp_path, 2000)

    observed.assert_not_called()


def test_update_rename_rebuilds_new_path(tmp_path: Path) -> None:
    """Remove old provenance and chunk the renamed source."""
    source = tmp_path / "old.txt"
    source.write_text("Unchanged content.\n", encoding="utf-8")
    previous_manifest = build_manifest(tmp_path, tmp_path, 10)
    previous_index = build_index(tmp_path, tmp_path, 10)
    source.rename(tmp_path / "new.txt")
    current_manifest = build_manifest(tmp_path, tmp_path, 10)

    with patch("src.incremental.chunk_file", wraps=chunk_file) as observed:
        updated = update_chunk_index(
            previous_index, previous_manifest, current_manifest, tmp_path,
        )

    assert [call.args[0] for call in observed.call_args_list] == ["new.txt"]
    assert updated == build_index(tmp_path, tmp_path, 10)


def test_update_reuses_empty_file_without_chunks(tmp_path: Path) -> None:
    """An unchanged empty file needs no entry in the old chunk list."""
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("Hello", encoding="utf-8")
    manifest = build_manifest(tmp_path, tmp_path)
    index = build_index(tmp_path, tmp_path)

    with patch("src.incremental.chunk_file", wraps=chunk_file) as observed:
        updated = update_chunk_index(index, manifest, manifest, tmp_path)

    observed.assert_not_called()
    assert updated == index


def test_update_rejects_removing_all_indexable_text(tmp_path: Path) -> None:
    """Keep the full indexer's policy against empty indexes."""
    source = tmp_path / "only.txt"
    source.write_text("Hello", encoding="utf-8")
    previous_manifest = build_manifest(tmp_path, tmp_path)
    previous_index = build_index(tmp_path, tmp_path)
    source.unlink()
    current_manifest = build_manifest(tmp_path, tmp_path)

    with pytest.raises(ValueError, match="no indexable text"):
        update_chunk_index(
            previous_index, previous_manifest, current_manifest, tmp_path,
        )
