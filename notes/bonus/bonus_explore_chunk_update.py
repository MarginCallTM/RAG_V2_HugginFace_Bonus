"""Compare incremental chunk updates with a full reconstruction."""


from pathlib import Path
from tempfile import TemporaryDirectory

from src.incremental import update_chunk_index, map_reusable_chunks
from src.indexer import build_index
from src.manifest import (
    build_manifest,
    compare_manifest,
    requires_full_rebuild,
)


def main() -> None:
    """Apply corpus changes and compare the resulting chunk indexes."""

    project_root = Path.cwd()
    experiments_dir = (
        project_root / "data/output/experiments/incremental"
    )
    experiments_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=experiments_dir) as directory:
        raw_dir = Path(directory)
        max_chunk_size = 20

        initial_files = {
            "cache.txt": "Reduce the memory allocated to the cache.\n",
            "server.txt": "The HTTP server listens on port 8000.\n",
            "old.txt": "Outdated documentation.\n",
        }

        for name, text in initial_files.items():
            (raw_dir / name).write_text(text, encoding="utf-8")

        previous_manifest = build_manifest(
            raw_dir, project_root, max_chunk_size,
        )
        previous_index = build_index(
            raw_dir, project_root, max_chunk_size,
        )

        (raw_dir / "server.txt").write_text(
            "The HTTP server listens on port 9000.\n",
            encoding="utf-8",
        )
        (raw_dir / "old.txt").unlink()
        (raw_dir / "aaa_guide.txt").write_text(
            "Read the installation guide.\n",
            encoding="utf-8",
        )

        current_manifest = build_manifest(
            raw_dir, project_root, max_chunk_size,
        )

        changes = compare_manifest(
            previous_manifest, current_manifest,
        )

        incremental_index = update_chunk_index(
            previous_index,
            previous_manifest,
            current_manifest,
            project_root,
        )

        full_index = build_index(
            raw_dir, project_root, max_chunk_size,
        )

        print("\nDetected changes:")
        print(changes.model_dump_json(indent=2))

        print("\nPrevious chunks:", len(previous_index.chunks))
        print("Incremental chunks:", len(incremental_index.chunks))
        print("Full rebuild chunks", len(full_index.chunks))
        print(
            "Same index:",
            incremental_index == full_index,
        )

        reusable_paths = set(changes.unchanged)

        if requires_full_rebuild(previous_manifest, current_manifest):
            reusable_paths.clear()

        old_to_new = map_reusable_chunks(
            previous_index,
            incremental_index,
            reusable_paths,
        )

        print("\nReusable chunk positions:", old_to_new)


if __name__ == "__main__":
    main()
