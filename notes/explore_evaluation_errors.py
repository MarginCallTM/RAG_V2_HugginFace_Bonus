"""Observe why duplicate result IDs must be rejected."""

from src.evaluation import evaluate_recall
from src.models import (
    AnsweredDataset,
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def main() -> None:
    """Compare valid results with results containing a duplicate ID."""
    source = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    dataset = AnsweredDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="q1",
                question="Example question",
                sources=[source],
                answer="Reference answer",
            )
        ]
    )
    found = MinimalSearchResults(
        question_id="q1",
        question="Example question",
        retrieved_sources=[source],
    )
    missing = MinimalSearchResults(
        question_id="q1",
        question="Example question",
        retrieved_sources=[],
    )

    valid = StudentSearchResults(search_results=[found], k=5)
    scores, mean_recall = evaluate_recall(dataset, valid)
    print("Resultats valides :", scores)
    print(f"Recall moyen : {mean_recall:.0%}")

    ambiguous = StudentSearchResults(
        search_results=[found, missing],
        k=5,
    )

    print("\nDeux resultats pour q1 :")
    print("Le premier retrouve la source ; le second ne trouve rien.")

    unsafe_lookup = {
        result.question_id: result
        for result in ambiguous.search_results
    }
    print("Nombre de resultats avant dictionnaire :", 2)
    print("Nombre de cles apres dictionnaire :", len(unsafe_lookup))
    print("La seconde entree a remplace la premiere.")

    try:
        evaluate_recall(dataset, ambiguous)
    except ValueError as error:
        print("\nNotre evaluateur refuse cette ambiguite :", error)


if __name__ == "__main__":
    main()
