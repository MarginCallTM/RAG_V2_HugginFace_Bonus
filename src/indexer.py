"""Build and persist the corpus chunk index."""

from pathlib import Path

from tqdm import tqdm

from src.chunking import chunk_file
from src.ingest import list_corpus_files, read_text, to_corpus_path
from src.models import Chunk, ChunkIndex


def build_index(
    raw_dir: Path,
    project_root: Path,
    max_chunk_size: int = 2000,
) -> ChunkIndex:
    """Build a validated chunk index from the candidate corpus files."""
    ChunkIndex(max_chunk_size=max_chunk_size, chunks=[])

    files = list_corpus_files(raw_dir)
    chunks: list[Chunk] = []

    for path in tqdm(files, desc="Chunking", unit="file"):
        source_path = to_corpus_path(path, project_root)
        text = read_text(path)

        chunks.extend(
            chunk_file(
                source_path,
                text,
                max_chunk_size,
            )
        )
    if not chunks:
        raise ValueError("Corpus contrains no indexable text")

    return ChunkIndex(
        max_chunk_size=max_chunk_size,
        chunks=chunks,
    )


def save_index(index: ChunkIndex, processed_dir: Path) -> Path:
    """Save an index as UTF-8 using a temporary file."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    target = processed_dir / "chunks.json"
    temporary = processed_dir / "chunks.json.tmp"

    serialized = index.model_dump_json()

    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(serialized)

    temporary.replace(target)

    return target


def load_index(processed_dir: Path) -> ChunkIndex:
    """Read and validate a chunk index saved on disk."""
    target = processed_dir / "chunks.json"

    with target.open(encoding="utf-8") as handle:
        serialized = handle.read()

    return ChunkIndex.model_validate_json(serialized)
