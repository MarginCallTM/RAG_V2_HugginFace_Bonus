"""Check chunking invariants across the complete candidate corpus."""

from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from src.chunking import chunk_file
from src.ingest import list_corpus_files, read_text, to_corpus_path


def check_chunks(
    source_path: str,
    text: str,
    max_chunk_size: int,
) -> int:
    """Chunk one document and verify its content, bounds and provenance"""
    chunks = chunk_file(source_path, text, max_chunk_size)
    expected_start = 0

    for chunk in chunks:
        start = chunk.first_character_index
        end = chunk.last_character_index

        assert chunk.file_path == source_path, "Wrong source path"
        assert start == expected_start, "Gap or overlap between chunks"
        assert 0 <= start < end <= len(text), "Invalid bounds"
        assert end - start <= max_chunk_size, "Chunk exceeds size limit"
        assert chunk.text == text[start:end], "Text does not match bounds"

        expected_start = end

    assert expected_start == len(text), "Document end is missing"
    assert "".join(chunk.text for chunk in chunks) == text, (
        "Recontructed document differs from original"
    )

    return len(chunks)


def main() -> None:
    """Process every candidate file and report failures and elapsed time."""
    project_root = Path.cwd()
    raw_dir = project_root / "data" / "raw" / "vllm-0.10.1"
    max_chunk_size = 2000

    started = perf_counter()
    files = list_corpus_files(raw_dir)

    total_chunks = 0
    successful_files = 0
    failures: list[tuple[str, str]] = []

    for path in tqdm(files, desc="Checking corpus", unit="file"):
        source_path = to_corpus_path(path, project_root)

        try:
            text = read_text(path)
            count = check_chunks(source_path, text, max_chunk_size)
        except Exception as error:
            failures.append(
                (source_path, f"{type(error).__name__}: {error}")
            )
            continue

        successful_files += 1
        total_chunks += count

    elapsed = perf_counter() - started

    print("\nTAILLE MAXIMALE :", max_chunk_size)
    print("FICHIER CANDIDATS :", len(files))
    print("FICHIER VALIDE :", successful_files)
    print("FICHIER EN ECHEC :", len(failures))
    print("CHUNKS DES FICHIERS VALIDES :", total_chunks)
    print("DUREE :", f"{elapsed:.2f} secondes")

    for source_path, message in failures:
        print("\nECHEC :", source_path)
        print(message)


if __name__ == "__main__":
    main()
