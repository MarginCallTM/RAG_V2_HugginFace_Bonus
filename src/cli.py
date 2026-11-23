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
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from src.datasets import (
    load_dataset,
    save_search_results,
    search_questions,
    answer_questions,
)
from src.retriever import Retriever
from src.retrieval import load_retriever
from src.evaluation import evaluate_recall
from src.prompting import build_budgeted_prompt


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

    def index_semantic(
        self,
        processed_dir: str = "data/processed",
        output_path: str = "data/processed/semantic_index.pt",
        batch_size: int = 32,
        max_tokens: int = 256,
    ) -> None:
        """Build a CPU semantic index from existing corpus chunks.

        Args:
            processed_dir: Directory containing chunks.json.
            output_path: Destination semantic index file.
            batch_size: Number of segments encoded per batch.
            max_tokens: Maximum tokens per segment, including special tokens.
        """
        started = perf_counter()

        try:
            from src.embeddings import EmbeddingConfig
            from src.semantic_index import (
                build_semantic_index,
                save_semantic_index,
            )

            config = EmbeddingConfig(
                batch_size=batch_size,
                max_tokens=max_tokens,
            )
            directory = Path(str(processed_dir))
            target = Path(str(output_path))

            if target.resolve() == (directory / "chunks.json").resolve():
                raise ValueError(
                    "Semantic output must not overwrite the chunk index."
                )

            index = load_index(directory)
            print(f"Loaded {len(index.chunks)} chunks", flush=True)
            print("Building semantic index on CPU...", flush=True)

            data, embeddings = build_semantic_index(index, config)
            saved_path = save_semantic_index(data, embeddings, target)
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            raise CliError(
                f"Semantic indexing failed: {error}"
            ) from error
        print(f"Saved semantic index to {saved_path}")
        print(f"Completed in {perf_counter() - started:.2f} seconds")

    def search(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        output_path: str = "data/output/search_results/single_query.json",
        method: str = "bm25",
        semantic_index_path: str = "data/processed/semantic_index.pt",
    ) -> None:
        """Search one question and save ranked source locxations.

        Args:
            query: Question to search for.
            k: Maximum number of returned sources.
            processed_dir: Directory containing the saved chunk index.
            output_path: Destination JSON file.
            method: Retrieval method: BM25 or semantic
            semantic_index_path: Saved semantic index files.
        """
        if type(k) is not int or k < 0:
            raise CliError("k must be a non-negative integer")

        question = UnansweredQuestion(question=str(query))

        try:
            retriever = load_retriever(
                method=str(method),
                processed_dir=Path(str(processed_dir)),
                semantic_index_path=Path(str(semantic_index_path)),
            )
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
        except (OSError, ValueError, RuntimeError, ImportError) as error:
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
        method: str = "bm25",
        semantic_index_path: str = "data/processed/semantic_index.pt"
    ) -> None:
        """Search a question dataset and save ranked source locations.

        Args:
            dataset_path: Input question dataset JSON file.
            k: Maximum number of sources returned per question.
            save_directory: Directory receiving the results JSON files.
            processed_dir: Directory containing the saved chunk index.
            method: Retrieval method: bm25 or semantic.
            semantic_index_path: Saved semantic index file.
        """
        if type(k) is not int or k < 0:
            raise CliError("k must be a non-negative integer")

        started = perf_counter()

        try:
            dataset_file = Path(str(dataset_path))
            dataset = load_dataset(dataset_file)

            retriever = load_retriever(
                method=str(method),
                processed_dir=Path(str(processed_dir)),
                semantic_index_path=Path(semantic_index_path),
            )

            results = search_questions(dataset, retriever, k=k)

            target = (
                Path(str(save_directory)) / dataset_file.name
            )
            save_search_results(results, target)
        except (OSError, ValueError, RuntimeError, ImportError) as error:
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

    def answer(
        self,
        query: str,
        k: int = 5,
        processed_dir: str = "data/processed",
        output_path: str = (
            "data/output/search_results_and_answer/single_query.json"
        ),
        model_name: str = "Qwen/Qwen3-0.6B",
        context_budget: int = 4096,
        max_new_tokens: int = 256,
    ) -> None:
        """Generate a grounded answer and save its included sources.

        Args:
            query: Question to answer.
            k: Maximum number of retrieved passages.
            processed_dir: Directory containing the saved chunk index.
            output_path: Destination JSON file.
            model_name: Hugging Face model identifier or local model path
            context_budget: Total input and output token budget
            max_new_tokens: Maximum number of generated tokens.
        """
        if type(k) is not int or k < 0:
            raise CliError("k must be a non-negative integer")

        if not str(query).strip():
            raise CliError("query must not be empty")

        if type(context_budget) is not int or context_budget <= 0:
            raise CliError("context_budget must be a positive integer")

        if (
            type(max_new_tokens) is not int
            or not 0 < max_new_tokens < context_budget
        ):
            raise CliError(
                "max_new_tokens must be positive and below context_budget"
            )

        started = perf_counter()

        try:
            from src.generator import Generator

            question = UnansweredQuestion(question=str(query))

            index = load_index(Path(str(processed_dir)))
            retriever = Retriever(index)
            hits = retriever.search(question.question, k=k)
            chunks = [hit.chunk for hit in hits]

            if chunks:
                print("Loading Qwen...", flush=True)
                generator = Generator(
                    model_name=str(model_name),
                    context_budget=context_budget,
                    max_new_tokens=max_new_tokens,
                )

                prompt = build_budgeted_prompt(
                    question.question,
                    chunks,
                    generator.tokenizer,
                    context_budget=generator.context_budget,
                    max_new_tokens=generator.max_new_tokens,
                )

                sources = prompt.sources

                if sources:
                    print("Generating answer...", flush=True)
                    answer_text = generator.generate(prompt)
                else:
                    answer_text = (
                        "Insufficient evidence: no retrieved passage "
                        "fits the input token budget."
                    )
            else:
                sources = []
                answer_text = (
                    "Insufficient evidence: no relevant passage "
                    "was retrieved"

                )

            output = StudentSearchResultsAndAnswer(
                search_results=[
                    MinimalAnswer(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=sources,
                        answer=answer_text,
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

        except (OSError, ValueError, RuntimeError, ImportError) as error:
            raise CliError(f"Answer generation failed: {error}") from error

        print("\nANSWER")
        print(answer_text)
        print(f"\nIncluded sources: {len(sources)}")
        print(f"Saved answer to {target}")
        print(f"Completed in {perf_counter() - started:.2f} seconds")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = (
            "data/output/search_results_and_answer/UnansweredQuestions"
        ),
        project_root: str = ".",
        model_name: str = "Qwen/Qwen3-0.6B",
        context_budget: int = 4096,
        max_new_tokens: int = 256,
        device: str = "auto",
    ) -> None:
        """Generate and save answers for existing search results

        Args:
            student_search_results_path: Input search results JSON.
            save_directory: Directory receiving the answers JSON.
            project_root: Root used to locate referebced corpus files.
            model_name: Model identifier or local model path.
            context_budget: Total input and output token budget.
            max_new_tokens: Maximum generated tokens per question.
            device: Execution device: auto, cpu or cuda
        """
        if type(context_budget) is not int or context_budget <= 0:
            raise CliError("context_budget must be a positive integer")
        if (
            type(max_new_tokens) is not int
            or not 0 < max_new_tokens < context_budget
        ):
            raise CliError(
                "max_new_token must be positive and below context_budget"
            )

        if device not in ("auto", "cpu", "cuda"):
            raise CliError("device must be auto, cpu or cuda")

        started = perf_counter()

        try:
            source_path = Path(str(student_search_results_path))
            target = Path(str(save_directory)) / source_path.name
            root = Path(str(project_root)).resolve()

            if source_path.resolve() == target.resolve():
                raise ValueError("Output must not overwrite search results")

            if not root.is_dir():
                raise ValueError("project_root must be an existing directory")

            with source_path.open(encoding="utf-8") as handle:
                results = StudentSearchResults.model_validate_json(
                    handle.read()
                )

            if results.k < 0:
                raise ValueError("k must be non negative")

            seen_ids: set[str] = set()

            for result in results.search_results:
                if result.question_id in seen_ids:
                    raise ValueError("Duplicate question_id in search results")

                seen_ids.add(result.question_id)

                if not result.question.strip():
                    raise ValueError("Question must not be empty")

                if len(result.retrieved_sources) > results.k:
                    raise ValueError("A question has more than k sources")

            total_questions = len(results.search_results)
            print(f"Loaded {total_questions} questions", flush=True)

            if results.search_results:
                from src.generator import Generator

                print("Loading Qwen...", flush=True)
                generator = Generator(
                    model_name=str(model_name),
                    context_budget=context_budget,
                    max_new_tokens=max_new_tokens,
                    device=str(device),
                )

                print(f"Device: {generator.device}", flush=True)

                output = answer_questions(results, generator, root)
            else:
                output = StudentSearchResultsAndAnswer(
                    search_results=[],
                    k=results.k,
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp")

            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(output.model_dump_json(indent=2))

            temporary.replace(target)
        except (OSError, ValueError, RuntimeError, ImportError) as error:
            raise CliError(
                f"Dataset answer generation failed: {error}"
            ) from error

        print(
            f"Processed {len(output.search_results)} "
            f"of {total_questions} questions"
        )
        print(f"Saved student_search_results_and_answer to {target}")
        print(f"Completed in {perf_counter() - started:.2f} seconds")
