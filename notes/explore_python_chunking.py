"""Observe Python chunking with different size limits"""

import json
from pathlib import Path

from src.chunking import split_python
from src.ingest import read_text


def main() -> None:
    """Load a referenced Python file and inspect overlapping chunks"""
    dataset_path = Path(
        "data/datasets/AnsweredQuestions/dataset_code_public.json"
    )

    with dataset_path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)

    question = dataset["rag_questions"][0]
    source = question["sources"][0]

    path = Path(source["file_path"])
    code = read_text(path)

    reference_start = source["first_character_index"]
    reference_end = source["last_character_index"]

    print("QUESTION :", question["question"])
    print("FICHIER :", path)
    print("LONGUEUR DU FICHIER :", len(code))
    print("BORNES DE REFERENCE :", reference_start, reference_end)

    print("\nPASSAGE DE REFERENCE :")
    print(code[reference_start:reference_end])

    for limit in (2000, 500):
        chunks = split_python(
            file_path=path.as_posix(),
            text=code,
            max_chunk_size=limit,
        )

        assert "".join(chunk.text for chunk in chunks) == code

        print("\n" + "=" * 50)
        print("TAILLE MAXIMAL :", limit)
        print("NOMBRE DE CHUNKS :", len(chunks))

        for number, chunk in enumerate(chunks, start=1):
            start = chunk.first_character_index
            end = chunk.last_character_index

            assert chunk.text == code[start:end]
            assert 0 < len(chunk.text) <= limit

            if start < reference_end and end > reference_start:
                print(f"\nCHUNK {number} [{start}:{end}]")
                print("LONGUEUR :", len(chunk.text))
                print(chunk.text)


if __name__ == "__main__":
    main()
