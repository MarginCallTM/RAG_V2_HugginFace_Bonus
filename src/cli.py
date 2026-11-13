"""Command-line interface for the RAG application."""

from pathlib import Path
from time import perf_counter

from src.indexer import build_index, save_index, load_index
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    UnansweredQuestion,
)
from src.retriever import Retriever


class CliError(Exception):
    """Represent an expected error reported to the CLI user."""


class RagCli:
    """Expose application commands."""

    def hello(self, name: str) -> None:
        """Display greeting"""
        print(f"Hello {name}!")

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ) -> None:
        """Build and save the corpus index.

        Args:
            max_chunk_size: Maximum characters per chunk, from 1 to 2000.
            raw_dir: Directory containing the source corpus.
            processed_dir: Directory receiving the saved index.
        """
        if type(max_chunk_size) is not int:
            raise CliError("max_chunk_size must be an integer")

        if not 1 <= max_chunk_size <= 2000:
            raise CliError("max_chunk_size must be between 1 and 2000")

        started = perf_counter()

        try:
            result = build_index(
                raw_dir=Path(str(raw_dir)),
                project_root=Path.cwd(),
                max_chunk_size=max_chunk_size,
            )
            target = save_index(result, Path(str(processed_dir)))
        except (OSError, ValueError) as error:
            raise CliError(f"Indexing failed: {error}") from error

        elapsed = perf_counter() - started

        print(f"Saved {len(result.chunks)} chunks to {target}")
        print(f"Indexing completed in {elapsed:.2f} seconds")

    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        output_path: str = "data/output/search_results/single_query.json",
    ) -> None:
        """Search one question and save ranked source locxations.

        Args:
            query: Question to search for.
            k: Maximum number of returned sources.
            processed_dir: Directory containing the saved chunk index.
            output_path: Destination JSON file.
        """
        if type(k) is not int or k < 0:
            raise CliError("k must be a non-negative integer")

        question = UnansweredQuestion(question=str(query))

        try:
            index = load_index(Path(str(processed_dir)))
            retriever = Retriever(index)
            results = retriever.search(question.question, k=k)

            sources = [
                MinimalSource(
                    file_path=result.chunk.file_path,
                    first_character_index=(
                        result.chunk.first_character_index
                    ),
                    last_character_index=(
                        result.chunk.last_character_index
                    ),
                )
                for result in results
            ]

            output = StudentSearchResults(
                search_results=[
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=sources,
                    )
                ],
                k=k,
            )

            target = Path(str(output_path))
            target.parent.mkdir(parents=True, exist_ok=True)

            temporary = target.with_name(target.name + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(output.model_dump_json(indent=2))

            temporary.replace(target)
        except (OSError, ValueError) as error:
            raise CliError(f"Search failed: {error}") from error

        if not results:
            print("No result")

        for rank, result in enumerate(results, start=1):
            chunk = result.chunk

            print(
                f"{rank}. {chunk.file_path} "
                f"[{chunk.first_character_index}:"
                f"{chunk.last_character_index}] "
                f"score={result.score:.3f}"
            )
        print(f"Saved search results to {target}")
