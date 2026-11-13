"""Data contracts exchanged between the RAG pipeline stages."""

from pydantic import BaseModel, model_validator, Field
import uuid


class MinimalSource(BaseModel):
    """Identify a character range in a corpus file."""

    file_path: str
    first_character_index: int
    last_character_index: int

    @model_validator(mode="after")
    def check_range(self) -> "MinimalSource":
        """Reject negative, empty or reversed character ranges."""
        if self.first_character_index < 0:
            raise ValueError("first_character_index must be >= 0")
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                "last_character_index must be > first_character_index"
            )

        return self


class UnansweredQuestion(BaseModel):
    """Represent a question without reference sources or answer."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with reference source and answer."""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represent a dataset of answered or unanswered questions"""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class AnsweredDataset(BaseModel):
    """Require reference sources and an answer for every question."""

    rag_questions: list[AnsweredQuestion]


class MinimalSearchResults(BaseModel):
    """Represent the sources retrived for one question."""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class StudentSearchResults(BaseModel):
    """Represent retrieval results for a dataset."""

    search_results: list[MinimalSearchResults]
    k: int


class MinimalAnswer(MinimalSearchResults):
    """Represent retrieved sources and a generated answer"""

    answer: str


class StudentSearchResultsAndAnswer(BaseModel):
    """Represent generated answers for a dataset."""

    search_results: list[MinimalAnswer]
    k: int


class Chunk(MinimalSource):
    """Represent a source passage and its exact text."""

    text: str

    @model_validator(mode="after")
    def check_text_lenght(self) -> "Chunk":
        """Require the texte lenght to match the character range."""
        expected_lenght = (
            self.last_character_index - self.first_character_index
        )

        if len(self.text) != expected_lenght:
            raise ValueError(
                "text length must match the character range"
            )
        return self


class ChunkIndex(BaseModel):
    """Store corpus chunks together with their size configuration."""

    max_chunk_size: int = Field(ge=1, le=2000)
    chunks: list[Chunk]

    @model_validator(mode="after")
    def check_chunk_sizes(self) -> "ChunkIndex":
        """Require every chunk to respect the stored size limits."""
        for chunk in self.chunks:
            if len(chunk.text) > self.max_chunk_size:
                raise ValueError(
                    "chunk exceeds the stored max_chunk_size"
                )
        return self


class ScoredChunk(BaseModel):
    """Associate a retrieved chunk with its lexical relevance score."""

    chunk: Chunk
    score: float
