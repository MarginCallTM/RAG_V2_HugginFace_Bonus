"""Explore dataset loading with Pydantic."""

from pathlib import Path

from src.models import RagDataset


def main() -> None:
    """Load both documentation datasets and inspect their first question."""
    root = Path("data/datasets")

    for scope in ("UnansweredQuestions", "AnsweredQuestions"):
        path = root / scope / "dataset_docs_public.json"

        with path.open(encoding="utf-8") as handle:
            dataset = RagDataset.model_validate_json(handle.read())

        first = dataset.rag_questions[0]
        second = dataset.rag_questions[1]

        print("\nDATASET :", scope)
        print("NOMBRE :", len(dataset.rag_questions))
        print("TYPE :", type(first).__name__)
        print("ID :", first.question_id)
        print("ID 2:", second.question_id)
        print("QUESTION :", first.question)
        print("QUESTION 2:", second.question)


if __name__ == "__main__":
    main()
