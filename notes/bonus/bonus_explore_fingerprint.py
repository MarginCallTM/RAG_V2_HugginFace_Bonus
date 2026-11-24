"""Explore content fingerprints for incremental indexing"""

from pathlib import Path
from pydantic import ValidationError

from src.ingest import read_text
from src.manifest import FileFingerprint, CorpusManifest
from src.manifest import (
    CorpusManifest,
    FileFingerprint,
    fingerprint,
)


def main() -> None:
    """Compare fingerprints before and after editing a simple file."""
    path = Path(
        "data/output/experiments/incremental/sample.txt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        "The HTTP server listens on port 8000.\n",
        encoding="utf-8",
    )

    original_hash = fingerprint(read_text(path))
    unchanged_hash = fingerprint(read_text(path))

    path.write_text(
        "The HTTP server listens on port 9000.\n",
        encoding="utf-8",
    )

    modified_hash = fingerprint(read_text(path))

    print("File:", path.as_posix())
    print("Original:", original_hash)
    print("Unchanged:", unchanged_hash)
    print("Modified:", modified_hash)

    print("\nSame content:", original_hash == unchanged_hash)
    print("Changed content:", original_hash != modified_hash)

    file_state = FileFingerprint(
        file_path=path.as_posix(),
        content_hash=modified_hash,
    )

    print("\nFile fingerprint:")
    print(file_state.model_dump_json(indent=2))

    manifest = CorpusManifest(max_chunk_size=2000, files=[file_state])

    print("\nCorpus manifest:")
    print(manifest.model_dump_json(indent=2))

    print("\nChecking duplicate path rejection...")

    try:
        CorpusManifest(max_chunk_size=2000, files=[file_state, file_state])
    except ValidationError as error:
        print("Duplicate rejected:", error)
    else:
        print("ERROR: duplicate path accepted")


if __name__ == "__main__":
    main()
