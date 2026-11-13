"""Observe question matching and mean recall on a small dataset."""

from src.evaluation import evaluate_recall
from src.models import (
    AnsweredDataset,
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def main() -> None:
    """Evaluate reordered results, then remove one question."""
    sources = {
        name: MinimalSource(
            file_path=f"data/raw/{name}.py",
            first_character_index=100,
            last_character_index=200,
        )
        for name in ("a", "b", "c", "d")
    }

    dataset = AnsweredDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="q1",
                question="Question 1",
                sources=[sources["a"]],
                answer="Reference 1",
            ),
            AnsweredQuestion(
                question_id="q2",
                question="Question 2",
                sources=[sources["b"], sources["c"]],
                answer="Reference 2",
            ),
            AnsweredQuestion(
                question_id="q3",
                question="Question 3",
                sources=[sources["d"]],
                answer="Reference 3",
            ),
        ]
    )

    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id=question_id,
                question=f"Question {question_id}",
                retrieved_sources=[sources[name]],
            )
            for question_id, name in (
                ("q3", "d"),
                ("q1", "a"),
                ("q2", "b"),
            )
        ],
        k=5,
    )

    print("Ordre du corrige : q1, q2, q3")
    print("Ordre des resultats : q3, q1, q2")

    scores, mean_recall = evaluate_recall(dataset, results)

    for question_id, score in scores.items():
        print(f"{question_id} : {score:.1%}")
    print(f"Moyenne : {mean_recall:.1%}")

    print("\nSuppression du resultat de q3, sans modifier le corrige.")
    incomplete = StudentSearchResults(
        search_results=[
            result
            for result in results.search_results
            if result.question_id != "q3"
        ],
        k=5,
    )

    scores, mean_recall = evaluate_recall(dataset, incomplete)

    for question_id, score in scores.items():
        print(f"{question_id} : {score:.1%}")
    print(f"Moyenne : {mean_recall:.1%}")


if __name__ == "__main__":
    main()
