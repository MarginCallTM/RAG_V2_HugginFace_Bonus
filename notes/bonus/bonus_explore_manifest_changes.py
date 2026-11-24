
"""Explore file additions between two corpus inventories."""

from pathlib import Path
from tempfile import TemporaryDirectory

from src.manifest import (
    build_manifest,
    compare_manifest,
)


def main() -> None:
    """Detect a file created after the first inventory."""
    project_root = Path.cwd()
    experiments_dir = (
        project_root / "data/output/experiments/incremental"
    )
    experiments_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=experiments_dir) as directory:
        raw_dir = Path(directory)

        (raw_dir / "server.txt").write_text(
            "The HTTP server listens on port 8000.\n",
            encoding="utf-8",
        )

        previous = build_manifest(raw_dir, project_root)

        (raw_dir / "extra.txt").write_text(
            "A new documentation file.\n",
            encoding="utf-8",
        )

        current = build_manifest(raw_dir, project_root)
        changes = compare_manifest(previous, current)

        print(changes.model_dump_json(indent=2))

        (raw_dir / "server.txt").unlink()

        after_deletion = build_manifest(raw_dir, project_root)
        deletion_changes = compare_manifest(
            current,
            after_deletion,
        )

        print("\nChanges after deletion:")
        print(deletion_changes.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
