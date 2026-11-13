"""Load question datasets and collect batch retrieval results"""

from pathlib import Path

from tqdm import tqdm

from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.retriever import Retriever


def load_dataset(path: Path) -> RagDataset:
    """Read and validate a question dataset."""
    with path.open(encoding="utf-8") as handle:
        return RagDataset.model_validate_json(handle.read())


def search_questions(
    dataset: RagDataset,
    retriever: Retriever,
    k: int,
) -> StudentSearchResults:
    """Search every question using one already prepared retriever. """
    if type(k) is not int or k < 0:
        raise ValueError("k must be a non-negative integer")

    results: list[MinimalSearchResults] = []

    for question in tqdm(
        dataset.rag_questions,
        desc="Searching",
        unit="question",
    ):
        hits = retriever.search(question.question, k=k)

        sources = [
            MinimalSource(
                file_path=hit.chunk.file_path,
                first_character_index=hit.chunk.first_character_index,
                last_character_index=hit.chunk.last_character_index,
            )
            for hit in hits
        ]

        results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            )
        )
    return StudentSearchResults(
        search_results=results,
        k=k,
    )


def save_search_results(
    results: StudentSearchResults,
    target: Path,
) -> Path:
    """Validate output limits and save search results as JSON"""
    if results.k < 0:
        raise ValueError("A question has more than k sources")

    for result in results.search_results:
        if len(result.retrieved_sources) > results.k:
            raise ValueError("A question has more than k sources")

        for source in result.retrieved_sources:
            lenght = (
                source.last_character_index
                - source.first_character_index
            )

            if lenght > 2000:
                raise ValueError("A source exceeds 2000 characters")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(results.model_dump_json(indent=2))

    temporary.replace(target)

    return target
