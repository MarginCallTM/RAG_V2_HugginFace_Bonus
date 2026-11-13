"""Inspect the files selected for ingestion."""

import json

from collections import Counter
from pathlib import Path

from src.ingest import list_corpus_files


def main() -> None:
    """Display selected files and count their line-ending categories."""
    files = list_corpus_files(Path("data/raw/vllm-0.10.1"))

    print("FICHIERS RETENUS :", len(files))

    for path in files[:5]:
        print(path)

    line_endings: Counter[str] = Counter()

    for path in files:
        content = path.read_bytes()

        if b"\r\n" in content:
            line_endings["avec CRLF"] += 1
        elif b"\n" in content:
            line_endings["LF sans CRLF"] += 1
        else:
            line_endings["sans LF"] += 1

    print("\nFINS DE LIGNE :")
    for category, count in line_endings.items():
        print(category, ":", count)

    crlf_paths = {
        path.as_posix()
        for path in files
        if b"\r\n" in path.read_bytes()
    }

    print("\nFICHIERS AVEC CRLF :")
    for file_path in sorted(crlf_paths):
        print(file_path)

    for name in ("docs", "code"):
        dataset_path = Path(
            "data/datasets/AnsweredQuestions"
        ) / f"dataset_{name}_public.json"

        with dataset_path.open(encoding="utf-8") as handle:
            dataset = json.load(handle)

        print(f"\nRÉFÉRENCES VERS CES FICHIERS — {name} :")
        count = 0

        for question in dataset["rag_questions"]:
            for source in question["sources"]:
                if source["file_path"] in crlf_paths:
                    print("QUESTION :", question["question"])
                    print("SOURCE :", source)
                    count += 1

        print("NOMBRE :", count)


if __name__ == "__main__":
    main()
