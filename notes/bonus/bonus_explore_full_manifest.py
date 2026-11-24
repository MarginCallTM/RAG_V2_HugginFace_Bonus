"""Explore manifest construction on the full source corpus"""

from pathlib import Path
from time import perf_counter

from src.manifest import build_manifest


def main() -> None:
    """Build the corpus manifest and display a short summary."""
    project_root = Path.cwd()
    raw_dir = project_root / "data/raw/vllm-0.10.1"

    print("Scanning corpus...", flush=True)
    start = perf_counter()

    manifest = build_manifest(raw_dir, project_root)

    elapsed = perf_counter() - start

    print("Tracked files:", len(manifest.files))
    print(f"Elapsed time: {elapsed:.2f} seconds")

    print("\nFirst three entries:")
    for file_state in manifest.files[:10]:
        print(file_state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
