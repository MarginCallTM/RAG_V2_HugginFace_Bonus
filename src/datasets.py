"""Load question datasets and collect batch retrieval results"""

from pathlib import Path

from tqdm import tqdm

from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    Chunk,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from src.retrieval import SearchRetriever
from src.ingest import read_text
from typing import TYPE_CHECKING
from src.prompting import build_budgeted_prompt

if TYPE_CHECKING:
    from src.generator import Generator


def load_dataset(path: Path) -> RagDataset:
    """Read and validate a question dataset."""
    with path.open(encoding="utf-8") as handle:
        return RagDataset.model_validate_json(handle.read())


def search_questions(
    dataset: RagDataset,
    retriever: SearchRetriever,
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


def load_source_chunk(
    source: MinimalSource,
    project_root: Path,
) -> Chunk:
    """Read a saved source span using the corpus text convention.

    Args:
        source: File path and character positions from retrieval.
        project_root: Root used to resolve relative source paths.

    Returns:
        The source passage with its original provenance.

    Raises:
        ValueError: If the path or character range is invalid.
        OSError: if the source file cannot be read.
    """
    root = project_root.resolve()
    relative_path = Path(source.file_path)

    if relative_path.is_absolute():
        raise ValueError("Source path must be relative to project_root")

    path = (root / relative_path).resolve()

    if not path.is_relative_to(root):
        raise ValueError("Source path is outside project_root")

    first = source.first_character_index
    last = source.last_character_index

    if last - first > 2000:
        raise ValueError("Source exceeds 2000 characters")

    text = read_text(path)

    if last > len(text):
        raise ValueError(
            f"Source range exceeds file length: {source.file_path}"
        )

    return Chunk(
        file_path=source.file_path,
        first_character_index=first,
        last_character_index=last,
        text=text[first:last],
    )


def answer_questions(
    results: StudentSearchResults,
    generator: "Generator",
    project_root: Path,
) -> StudentSearchResultsAndAnswer:
    """Generate answers using one already loaded model.

    Args:
        results: Saved questions and ranked source locations.
        generator: Model instance reused across all questions.
        project_root: Root used to locate source files.

    Returns:
        Answers preserving question identifiers and included sources.

    Raises:
        ValueError: If question identifiers or source counts are invalid.
        OSError: If a referenced source file cannot be read
    """
    if results.k < 0:
        raise ValueError("k must be non-negative")

    question_ids = [
        result.question_id
        for result in results.search_results
    ]

    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Duplicate question_id in search results")

    for result in results.search_results:
        if not result.question.strip():
            raise ValueError("Question must not be empty")

        if len(result.retrieved_sources) > results.k:
            raise ValueError("A question has more than k sources")

    answers: list[MinimalAnswer] = []

    for result in tqdm(
        results.search_results,
        desc="Answering",
        unit="question",
    ):
        chunks = [
            load_source_chunk(source, project_root)
            for source in result.retrieved_sources
        ]

        sources: list[MinimalSource] = []

        if chunks:
            prompt = build_budgeted_prompt(
                result.question,
                chunks,
                generator.tokenizer,
                context_budget=generator.context_budget,
                max_new_tokens=generator.max_new_tokens,
            )

            sources = prompt.sources

            if sources:
                answer_text = generator.generate(prompt)
            else:
                answer_text = (
                    "Insufficient evidence: no retrieved passage "
                    "fits the input token budget."
                )
        else:
            answer_text = (
                "Insufficient evidence: no relevant passage "
                "was retrieved."
            )

        answers.append(
            MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=sources,
                answer=answer_text,
            )
        )

    return StudentSearchResultsAndAnswer(
        search_results=answers,
        k=results.k,
    )
