"""Observe batch retrieval over the public documentation question"""

from pathlib import Path
from time import perf_counter

from src.datasets import (
    load_dataset,
    save_search_results,
    search_questions,
)
from src.indexer import load_index
from src.retriever import Retriever


def main() -> None:
    """Prepare one retriever and search a complete question dataset."""
    started = perf_counter()

    dataset = load_dataset(
        Path(
            "data/datasets/UnansweredQuestions/"
            "dataset_docs_public.json"
        )
    )
    index = load_index(Path("data/processed"))
    retriever = Retriever(index)

    ready_at = perf_counter()
    output = search_questions(dataset, retriever, k=5)
    target = save_search_results(
        output,
        Path(
            "data/output/search_results/UnansweredQuestions/"
            "dataset_docs_public.json"
        )
    )
    finished_at = perf_counter()

    first = output.search_results[0]

    print("\nQUESTION TRAITEES :", len(output.search_results))
    print("K :", output.k)
    print("PREMIER ID CONSERVE :", first.question_id)
    print("PREMIERE QUESTION :", first.question)
    print("SOURCE RETROUVER :", len(first.retrieved_sources))

    for source in first.retrieved_sources:
        print(source.file_path)

    print("\nPREPARATION:", f"{ready_at - started:.2f} s")
    print("RECHERCHES :", f"{finished_at - ready_at:2f} s")
    print("TOTAL :", f"{finished_at - started:.2f} s")
    print("FICHIER SAUVGARDER :", target)


if __name__ == "__main__":
    main()
