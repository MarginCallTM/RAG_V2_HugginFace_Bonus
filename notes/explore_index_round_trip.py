"""Measure full-corpus index construction, persistence and loading"""

from pathlib import Path
from time import perf_counter

from src.indexer import build_index, load_index, save_index


def main() -> None:
    """Save and reload the complete corpus index"""
    project_root = Path.cwd()
    raw_dir = project_root / "data" / "raw" / "vllm-0.10.1"
    processed_dir = project_root / "data" / "processed"

    started = perf_counter()

    index = build_index(
        raw_dir=raw_dir,
        project_root=project_root,
        max_chunk_size=2000,
    )
    built_at = perf_counter()

    target = save_index(index, processed_dir)
    saved_at = perf_counter()

    restored = load_index(processed_dir)
    loaded_at = perf_counter()

    assert restored == index

    print("\nCHUNKS CONSTRUITS :", len(index.chunks))
    print("CHUNKS RECHARGES :", len(restored.chunks))
    print("DONNEES IDENTIQUES :", restored == index)
    print("FICHIER :", target)
    print("TAILLE :", target.stat().st_size, "octets")

    print("\nCONSTRUCTION :", f"{built_at - started:.2f} s")
    print("SAUVEGARDE :", f"{saved_at - built_at:.2f} s")
    print("PREPARATION COMPLETE :", f"{saved_at - started:2f} s")
    print("RECHARGEMENT :", f"{loaded_at - saved_at:.2f} s")


if __name__ == "__main__":
    main()
