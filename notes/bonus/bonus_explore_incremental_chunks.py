"""Explore rebuilding chunks from a verified file inventory"""

from pathlib import Path
from tempfile import TemporaryDirectory

from src.incremental import rebuild_file_chunks
from src.manifest import build_manifest


def main() -> None:
    """Build chunks for one file tracked by a manifest"""
    project_root = Path.cwd()
    experiments_dir = (
        project_root / "data/output/experiments/incremental"
    )
    experiments_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=experiments_dir) as directory:
        raw_dir = Path(directory)
        text = "The HTTP server listens on port 8000.\n"

        (raw_dir / "server.txt").write_text(
            text,
            encoding="utf-8",
        )

        manifest = build_manifest(
            raw_dir,
            project_root,
            max_chunk_size=20,
        )

        chunks = rebuild_file_chunks(
            manifest.files[0],
            project_root,
            manifest.max_chunk_size,
        )

        print("Chunks:", len(chunks))
        print(
            "Original text preserved:",
            "".join(chunk.text for chunk in chunks) == text,
        )

        for chunk in chunks:
            print(
                f"[{chunk.first_character_index}:"
                f"{chunk.last_character_index}] "
                f"{chunk.text!r}"
            )

        (raw_dir / "server.txt").write_text(
            "The HTTP server listens on port 9000.\n",
            encoding="utf-8",
        )
        print("\nChecking changes after inventory...")

        try:
            rebuild_file_chunks(
                manifest.files[0],
                project_root,
                manifest.max_chunk_size,
            )
        except ValueError as error:
            print("Rebuild rejected:", error)
        else:
            print("ERROR: Changed file accepted")


if __name__ == "__main__":
    main()
