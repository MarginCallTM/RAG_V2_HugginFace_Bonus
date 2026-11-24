"""Tests for corpus fingerprints, inventories and change detection."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.manifest import (
    CorpusManifest,
    FileFingerprint,
    ManifestDiff,
    build_manifest,
    compare_manifest,
    fingerprint,
    load_manifest,
    requires_full_rebuild,
    save_manifest,
)


def test_fingerprint_matches_known_sha256() -> None:
    """Use a known digest to verify the hashing contract."""
    assert fingerprint("abc") == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_fingerprint_detects_same_length_change() -> None:
    """Detect content changes even when the text length stays identical."""
    assert fingerprint("port 8000") != fingerprint("port 9000")


@pytest.mark.parametrize("digest", ["a" * 63, "a" * 65, "g" * 64])
def test_file_fingerprint_rejects_invalid_digest(digest: str) -> None:
    """Require the expected hexadecimal SHA-256 representation."""
    with pytest.raises(ValidationError):
        FileFingerprint(file_path="server.txt", content_hash=digest)


def test_file_fingerprint_rejects_empty_path() -> None:
    """Require an identity for each tracked file."""
    with pytest.raises(ValidationError):
        FileFingerprint(file_path="", content_hash=fingerprint("text"))


def test_manifest_rejects_conflicting_duplicate_paths() -> None:
    """Reject two different states for the same source path."""
    with pytest.raises(ValidationError, match="Duplicate file path"):
        CorpusManifest(max_chunk_size=2000, files=[
            FileFingerprint(
                file_path="server.txt", content_hash=fingerprint("8000"),
            ),
            FileFingerprint(
                file_path="server.txt", content_hash=fingerprint("9000"),
            ),
        ])


def test_manifest_allows_identical_content_at_different_paths() -> None:
    """File identity comes from its path rather than its digest."""
    manifest = CorpusManifest(max_chunk_size=2000, files=[
        FileFingerprint(file_path=path, content_hash=fingerprint("same"))
        for path in ("a.txt", "b.txt")
    ])

    assert len(manifest.files) == 2


def test_build_manifest_tracks_sorted_relative_paths(tmp_path: Path) -> None:
    """Discover supported sources recursively, including empty files."""
    raw_dir = tmp_path / "corpus"
    nested = raw_dir / "nested"
    nested.mkdir(parents=True)
    (nested / "server.py").write_text("pass\n", encoding="utf-8")
    (raw_dir / "empty.txt").write_text("", encoding="utf-8")
    (raw_dir / "cache.md").write_text("Cache été\n", encoding="utf-8")
    (raw_dir / "ignored.json").write_text("{}", encoding="utf-8")

    manifest = build_manifest(raw_dir, tmp_path)

    assert manifest == CorpusManifest(max_chunk_size=2000, files=[
        FileFingerprint(
            file_path="corpus/cache.md",
            content_hash=fingerprint("Cache été\n"),
        ),
        FileFingerprint(
            file_path="corpus/empty.txt", content_hash=fingerprint(""),
        ),
        FileFingerprint(
            file_path="corpus/nested/server.py",
            content_hash=fingerprint("pass\n"),
        ),
    ])


def test_build_manifest_normalizes_newlines(tmp_path: Path) -> None:
    """Fingerprint the normalized text used by the chunking pipeline."""
    path = tmp_path / "server.txt"
    path.write_bytes(b"first\r\nsecond\r\n")
    previous = build_manifest(tmp_path, tmp_path)
    path.write_bytes(b"first\nsecond\n")

    assert build_manifest(tmp_path, tmp_path) == previous


def test_build_manifest_handles_empty_directory(tmp_path: Path) -> None:
    """An empty inventory is valid even without indexable chunks."""
    assert build_manifest(tmp_path, tmp_path) == CorpusManifest(
        max_chunk_size=2000, files=[],
    )


def test_build_manifest_rejects_missing_directory(tmp_path: Path) -> None:
    """Do not interpret an unavailable corpus as mass file deletion."""
    with pytest.raises(NotADirectoryError):
        build_manifest(tmp_path / "missing", tmp_path)


def test_build_manifest_propagates_read_failure(tmp_path: Path) -> None:
    """Fail rather than returning an inventory that omits unreadable text."""
    (tmp_path / "a.txt").write_text("valid", encoding="utf-8")
    (tmp_path / "b.txt").write_bytes(b"\xff")

    with pytest.raises(UnicodeDecodeError):
        build_manifest(tmp_path, tmp_path)


def test_compare_manifest_without_changes(tmp_path: Path) -> None:
    """Repeated scans classify all sources as unchanged."""
    for name in ("b.txt", "a.md"):
        (tmp_path / name).write_text("same", encoding="utf-8")
    previous = build_manifest(tmp_path, tmp_path)

    current = build_manifest(tmp_path, tmp_path)
    changes = compare_manifest(previous, current)

    assert changes == ManifestDiff(
        unchanged=["a.md", "b.txt"], added=[], modified=[], deleted=[],
    )


def test_compare_manifest_classifies_all_changes(tmp_path: Path) -> None:
    """Classify additions, edits, deletions and unchanged files together."""
    for name in ("keep.txt", "edit.txt", "remove.txt"):
        (tmp_path / name).write_text("port 8000", encoding="utf-8")
    previous = build_manifest(tmp_path, tmp_path)
    previous_snapshot = previous.model_dump()

    (tmp_path / "edit.txt").write_text("port 9000", encoding="utf-8")
    (tmp_path / "remove.txt").unlink()
    for name in ("z_new.txt", "a_new.txt"):
        (tmp_path / name).write_text("new", encoding="utf-8")

    changes = compare_manifest(previous, build_manifest(tmp_path, tmp_path))

    assert changes == ManifestDiff(
        unchanged=["keep.txt"],
        added=["a_new.txt", "z_new.txt"],
        modified=["edit.txt"],
        deleted=["remove.txt"],
    )
    assert previous.model_dump() == previous_snapshot


def test_compare_manifest_treats_rename_as_delete_and_add(
    tmp_path: Path,
) -> None:
    """A new path is a new identity even if its content is unchanged."""
    source = tmp_path / "server.txt"
    source.write_text("port 8000", encoding="utf-8")
    previous = build_manifest(tmp_path, tmp_path)
    source.rename(tmp_path / "http.txt")

    changes = compare_manifest(previous, build_manifest(tmp_path, tmp_path))

    assert changes == ManifestDiff(
        unchanged=[], added=["http.txt"], modified=[],
        deleted=["server.txt"],
    )


def test_compare_empty_manifests() -> None:
    """Comparing two empty inventories produces no changes."""
    empty = CorpusManifest(max_chunk_size=2000, files=[])

    assert compare_manifest(empty, empty) == ManifestDiff(
        unchanged=[], added=[], modified=[], deleted=[],
    )


@pytest.mark.parametrize("size", [0, -1, 2001])
def test_manifest_rejects_invalid_chunk_size(size: int) -> None:
    """Reject chunk sizes outside the supported range."""
    with pytest.raises(ValidationError):
        CorpusManifest(max_chunk_size=size, files=[])


def test_manifest_requires_chunk_size() -> None:
    """Do not silently assume the previous chunking configuration."""
    with pytest.raises(ValidationError, match="max_chunk_size"):
        CorpusManifest.model_validate({"files": []})


def test_manifest_rejects_unsupported_format_version() -> None:
    """Reject a manifest format this program cannot read."""
    with pytest.raises(ValidationError, match="format_version"):
        CorpusManifest.model_validate({
            "format_version": 2,
            "max_chunk_size": 2000,
            "files": [],
        })


def test_manifest_rejects_unsupported_preparation_version() -> None:
    """Reject an unsupported preparation version."""
    with pytest.raises(ValidationError, match="preparation_version"):
        CorpusManifest.model_validate({
            "preparation_version": 2,
            "max_chunk_size": 2000,
            "files": [],
        })


def test_build_manifest_preserves_chunk_size(tmp_path: Path) -> None:
    """Record the chunk size supplied by the caller."""
    manifest = build_manifest(
        tmp_path,
        tmp_path,
        max_chunk_size=1000,
    )

    assert manifest.max_chunk_size == 1000


def test_manifest_save_and_load(tmp_path: Path) -> None:
    """Preserve file fingerprints and configuration through JSON."""
    manifest = CorpusManifest(
        max_chunk_size=1000,
        files=[
            FileFingerprint(
                file_path="corpus/server.txt",
                content_hash=fingerprint("port 8000"),
            ),
        ],
    )
    target = tmp_path / "processed" / "manifest.json"

    saved_path = save_manifest(manifest, target)
    restored = load_manifest(saved_path)

    assert saved_path == target
    assert restored == manifest
    assert not target.with_name("manifest.json.tmp").exists()


def test_load_manifest_rejects_missing_file(tmp_path: Path) -> None:
    """Report a missing reference inventory."""
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "missing.json")


def test_load_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    """Do not accept a corrupted JSON file."""
    target = tmp_path / "manifest.json"
    target.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_manifest(target)


def test_load_manifest_validates_stored_data(tmp_path: Path) -> None:
    """Reject valid JSON containing an invalid configuration."""
    target = tmp_path / "manifest.json"
    target.write_text(
        '{"max_chunk_size": 0, "files": []}',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="max_chunk_size"):
        load_manifest(target)


@pytest.mark.parametrize(
    ("current_size", "expected"),
    [(2000, False), (1000, True)],
)
def test_rebuild_depends_on_chunk_size(
    current_size: int,
    expected: bool,
) -> None:
    """Require rebuilding when the requested chunk size changes."""
    previous = CorpusManifest(max_chunk_size=2000, files=[])
    current = CorpusManifest(
        max_chunk_size=current_size,
        files=[],
    )

    assert requires_full_rebuild(previous, current) == expected
