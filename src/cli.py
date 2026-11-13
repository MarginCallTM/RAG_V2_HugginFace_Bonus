"""Command-line interface for the RAG application."""

from pathlib import Path
from time import perf_counter

from src.indexer import build_index, save_index, load_index
from src.models import (
    UnansweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    AnsweredDataset,
)
from src.datasets import (
    load_dataset,
    save_search_results,
    search_questions,
)
from src.retriever import Retriever
from src.evaluation import evaluate_recall


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

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 5,
        save_directory: str = (
            "data/output/search_results/UnansweredQuestions"
        ),
        processed_dir: str = "data/processed",
    ) -> None:
        """Search a question dataset and save ranked source locations.

        Args:
            dataset_path: Input question dataset JSON file.
            k: Maximum number of sources returned per question.
            save_directory: Directory receiving the results JSON files.
            processed_dir: Directory containing the saved chunk index.
        """
        if type(k) is not int or k < 0:
            raise CliError("k must be a non-negative integer")

        started = perf_counter()

        try:
            dataset_file = Path(str(dataset_path))
            dataset = load_dataset(dataset_file)

            index = load_index(Path(str(processed_dir)))
            retriever = Retriever(index)

            results = search_questions(dataset, retriever, k=k)

            target = (
                Path(str(save_directory)) / dataset_file.name
            )
            save_search_results(results, target)
        except (OSError, ValueError) as error:
            raise CliError(
                f"Dataset search failed: {error}"
            ) from error

        elapsed = perf_counter() - started

        print(f"Processed {len(results.search_results)} questions")
        print(f"Saved search result tp {target}")
        print(f"Dataset search completed in {elapsed:.2f} seconds")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 5,
    ) -> None:
        """Evaluate saved retrieval results against references sources.

        Args:
            student_search_results_path: Search results JSON file.
            dataset_path: Answered reference dataset JSON file.
            k: number of ranked results considered per question.
        """
        if type(k) is not int or k < 0:
            raise CliError("k must be a non-negative integer")

        try:
            reference_path = Path(str(dataset_path))
            results_path = Path(str(student_search_results_path))

            with reference_path.open(encoding="utf-8") as handle:
                dataset = AnsweredDataset.model_validate_json(
                    handle.read()
                )
            with results_path.open(encoding="utf-8") as handle:
                results = StudentSearchResults.model_validate_json(
                    handle.read()
                )

            scores, mean_recall = evaluate_recall(
                dataset,
                results,
                k=k,
            )
        except (OSError, ValueError) as error:
            raise CliError(f"Evaluation failed: {error}") from error

        result_ids = {
            result.question_id
            for result in results.search_results
        }
        missing_count = sum(
            question.question_id not in result_ids
            for question in dataset.rag_questions
        )
        zero_count = sum(
            score == 0.0
            for score in scores.values()
        )

        print(f"Evaluated {len(scores)} questions.")
        print(f"Questions missing from results: {missing_count}")
        print(f"Questions with zero recall:{zero_count}")
        print(f"Recall@{k}: {mean_recall:.2%}")
