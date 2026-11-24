"""Explore automatic manifest construction on a small corpus."""

from pathlib import Path

from src.manifest import build_manifest, compare_manifest


def main() -> None:
    """Create two sample files display their manifest"""
    project_root = Path.cwd()
    raw_dir = (
        project_root
        / "data/output/experiments/incremental/corpus"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "server.txt").write_text(
        "The HTTP server listens on port 8000.\n",
        encoding="utf-8",
    )

    (raw_dir / "cache.md").write_text(
        "Reduce the memory allocated to the cache.\n",
        encoding="utf-8",
    )

    manifest = build_manifest(raw_dir, project_root)

    print("Tracked files:", len(manifest.files))
    print(manifest.model_dump_json(indent=2))

    (raw_dir / "server.txt").write_text(
        "The HTTP server listens on port 9000.\n",
        encoding="utf-8",
    )

    current_manifest = build_manifest(raw_dir, project_root)
    changes = compare_manifest(manifest, current_manifest)

    print("\nDetected changes:")
    print(changes.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
