"""Validated requests and reusable CPU services for the local HTTP API."""

from pathlib import Path
from threading import Lock
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator
from transformers import PreTrainedTokenizerBase

from src.models import MinimalSource, ScoredChunk
from src.prompting import RagPrompt, build_budgeted_prompt
from src.retrieval import SearchRetriever, load_retriever


class ApiConfig(BaseModel):
    """Configure one index snapshot and CPU generation for a server."""

    method: Literal["bm25", "semantic", "hybrid"] = "bm25"
    processed_dir: Path = Path("data/processed")
    semantic_index_path: Path = Path("data/processed/semantic_index.pt")
    model_name: str = Field(default="Qwen/Qwen3-0.6B", min_length=1)
    context_budget: int = Field(default=4096, gt=0)
    max_new_tokens: int = Field(default=256, gt=0)


class QueryRequest(BaseModel):
    """Accept a question and a bounded number of retrieved chunks."""

    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=16000)
    k: int = Field(default=5, ge=0, le=20)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        """Preserve the question while rejecting whitespace-only input."""
        if not value.strip():
            raise ValueError("Question must not be empty")
        return value


class SearchResponse(BaseModel):
    """Return ranked chunks and method-specific scores."""

    question: str
    k: int
    method: str
    results: list[ScoredChunk]


class AnswerResponse(BaseModel):
    """Return the answer and only sources actually included in its prompt."""

    question: str
    k: int
    method: str
    answer: str
    sources: list[MinimalSource]


class AnswerGenerator(Protocol):
    """Describe the existing generator without requiring model loading."""

    tokenizer: PreTrainedTokenizerBase
    context_budget: int
    max_new_tokens: int

    def generate(self, prompt: RagPrompt) -> str:
        """Generate from an already budgeted prompt."""
        ...


class ApiError(Exception):
    """Carry an HTTP status and a message safe to return to the client."""

    def __init__(self, status: int, message: str) -> None:
        """Record the status and public error message."""
        super().__init__(message)
        self.status = status


class RagService:
    """Retain one retriever and lazily load one CPU answer generator."""

    def __init__(
        self,
        config: ApiConfig,
        retriever: SearchRetriever | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        """Load the index once; permit lightweight injected test doubles."""
        if config.max_new_tokens >= config.context_budget:
            raise ValueError("max_new_tokens must be below context_budget")
        self.config = config
        self.retriever = (
            retriever if retriever is not None else load_retriever(
                config.method, config.processed_dir,
                config.semantic_index_path,
            )
        )
        self.generator = generator
        self.work_lock = Lock()

    def _load_generator(self) -> AnswerGenerator:
        """Load Qwen on CPU once, while the work lock is held."""
        if self.generator is None:
            from src.generator import Generator

            try:
                self.generator = Generator(
                    model_name=self.config.model_name,
                    context_budget=self.config.context_budget,
                    max_new_tokens=self.config.max_new_tokens,
                    device="cpu",
                )
            except (OSError, ValueError, RuntimeError, ImportError) as error:
                raise ApiError(
                    503, "Answer model could not be loaded",
                ) from error
        return self.generator

    def execute(
        self, route: str, request: QueryRequest,
    ) -> SearchResponse | AnswerResponse:
        """Run one search or answer; reject overlapping expensive requests."""
        if not self.work_lock.acquire(blocking=False):
            raise ApiError(503, "Server busy; retry after the current request")
        try:
            try:
                hits = self.retriever.search(request.question, k=request.k)
            except ValueError as error:
                raise ApiError(422, str(error)) from error
            if route == "/search":
                return SearchResponse(
                    question=request.question, k=request.k,
                    method=self.config.method, results=hits,
                )
            if not hits:
                return AnswerResponse(
                    question=request.question, k=request.k,
                    method=self.config.method, sources=[],
                    answer=(
                        "Insufficient evidence: no relevant passage retrieved."
                    ),
                )
            generator = self._load_generator()
            try:
                prompt = build_budgeted_prompt(
                    request.question, [hit.chunk for hit in hits],
                    generator.tokenizer, generator.context_budget,
                    generator.max_new_tokens,
                )
            except ValueError as error:
                raise ApiError(422, str(error)) from error
            answer = (
                generator.generate(prompt) if prompt.sources else
                "Insufficient evidence: no passage fits the token budget."
            )
            return AnswerResponse(
                question=request.question, k=request.k,
                method=self.config.method, answer=answer,
                sources=prompt.sources,
            )
        finally:
            self.work_lock.release()
