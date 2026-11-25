"""Update corpus chunks while reusing unchanged file data."""

from pathlib import Path

from src.chunking import chunk_file
from src.ingest import read_text
from src.manifest import (
    FileFingerprint,
    fingerprint,
    CorpusManifest,
    compare_manifest,
    requires_full_rebuild,
)
from src.models import Chunk, ChunkIndex


def rebuild_file_chunks(
    file_state: FileFingerprint,
    project_root: Path,
    max_chunk_size: int,
) -> list[Chunk]:
    """Chunk a file only if its text matches the current inventory."""
    path = project_root / file_state.file_path
    text = read_text(path)

    if fingerprint(text) != file_state.content_hash:
        raise ValueError(
            f"File changed after inventory: {file_state.file_path}"
        )

    return chunk_file(
        file_state.file_path,
        text,
        max_chunk_size,
    )


def update_chunk_index(
    previous_index: ChunkIndex,
    previous_manifest: CorpusManifest,
    current_manifest: CorpusManifest,
    project_root: Path,
) -> ChunkIndex:
    """Reuse unchanged file chunks and rebuild the remaining files."""
    if previous_index.max_chunk_size != previous_manifest.max_chunk_size:
        raise ValueError(
            "Previous index and manifest have different chunk size"
        )

    previous_paths = {
        file.file_path for file in previous_manifest.files
    }

    chunks_by_path: dict[str, list[Chunk]] = {}

    for chunk in previous_index.chunks:
        if chunk.file_path not in previous_paths:
            raise ValueError(
                "Previous index contains a file absent from its manifest"
            )

        chunks_by_path.setdefault(chunk.file_path, []).append(chunk)

    changes = compare_manifest(previous_manifest, current_manifest)
    reusable_paths = set(changes.unchanged)

    if requires_full_rebuild(previous_manifest, current_manifest):
        reusable_paths.clear()

    chunks: list[Chunk] = []

    for file_state in sorted(
        current_manifest.files,
        key=lambda file: file.file_path,
    ):
        if file_state.file_path in reusable_paths:
            chunks.extend(
                chunks_by_path.get(file_state.file_path, [])
            )
        else:
            chunks.extend(
                rebuild_file_chunks(
                    file_state,
                    project_root,
                    current_manifest.max_chunk_size,
                )
            )
    if not chunks:
        raise ValueError("Corpus contains no indexable text")

    return ChunkIndex(
        max_chunk_size=current_manifest.max_chunk_size,
        chunks=chunks,
    )


def map_reusable_chunks(
    previous_index: ChunkIndex,
    current_index: ChunkIndex,
    reusable_paths: set[str],
) -> dict[int, int]:
    """Map old chunk positions to their new positions for reusable files."""
    previous_positions: dict[tuple[str, int, int], int] = {}

    for position, chunk in enumerate(previous_index.chunks):
        if chunk.file_path in reusable_paths:
            key = (
                chunk.file_path,
                chunk.first_character_index,
                chunk.last_character_index,
            )
            previous_positions[key] = position
    old_to_new: dict[int, int] = {}

    for new_position, chunk in enumerate(current_index.chunks):
        if chunk.file_path not in reusable_paths:
            continue

        key = (
            chunk.file_path,
            chunk.first_character_index,
            chunk.last_character_index,
        )

        if key not in previous_positions:
            raise ValueError("Reusable chunk missing from previous index")

        old_position = previous_positions[key]

        if previous_index.chunks[old_position] != chunk:
            raise ValueError("Reusable chunk content has changed")

        old_to_new[old_position] = new_position

    return old_to_new
