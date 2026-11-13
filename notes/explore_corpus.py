"""Inspect corpus extensions and reference-source extensions."""

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    """Print extension counts for the corpus and references sources."""
    corpus = Path("data/raw/vllm-0.10.1")
    print("DOSSIER COURANT :", Path.cwd())
    print("CHEMIN DU CORPUS :", corpus.resolve())
    print("DOSSIER EXISTANT :", corpus.is_dir())

    if not corpus.is_dir():
        raise FileNotFoundError(
            f"Corpus introuvable : {corpus.resolve()}"
        )
    extensions = Counter(
        path.suffix or "(sans extension)"
        for path in corpus.rglob("*")
        if path.is_file()
    )

    print("EXTENSION DU CORPUS :")
    for extension, count in extensions.most_common(15):
        print(f"{extension:20} {count}")

    for name in ("docs", "code"):
        dataset_path = Path(
            "data/datasets/AnsweredQuestions"
        ) / f"dataset_{name}_public.json"

        with dataset_path.open(encoding="utf-8") as handle:
            dataset = json.load(handle)

        source_extensions = Counter(
            Path(source["file_path"]).suffix or "(sans extension)"
            for question in dataset["rag_questions"]
            for source in question["sources"]
        )

        print(f"\nEXTENSIONS DES SOURCES — {name} :")
        for extension, count in source_extensions.most_common():
            print(f"{extension:20} {count}")

        if name == "docs":
            print("\nSOURCES TXT :")

            for question in dataset["rag_questions"]:
                for source in question["sources"]:
                    if Path(source["file_path"]).suffix == ".txt":
                        print("QUESTION :", question["question"])
                        print("FICHIER :", source["file_path"])
                        print()


if __name__ == "__main__":
    main()
