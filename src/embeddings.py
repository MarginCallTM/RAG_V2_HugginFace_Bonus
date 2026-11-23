"""Configure and load the CPU semantic encoder."""

from typing import cast

import torch
import torch.nn.functional as F
from tqdm import tqdm
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer
from src.models import Chunk


class EmbeddingConfig(BaseModel):
    """Describe the model and processing settings for embeddings."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
    )
    revision: str = Field(
        default="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        min_length=1
    )
    max_tokens: int = Field(default=256, ge=3, le=256)
    batch_size: int = Field(default=32, ge=1)


class TokenSegment(BaseModel):
    """Associate a token sequence with its original corpus chunk."""

    chunk_position: int = Field(ge=0)
    segment_position: int = Field(ge=0)
    input_ids: list[int] = Field(min_length=1)


class SemanticEncoder:
    """Load and retain a tokenizer and a model for CPU inference"""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Load the configured model once for this encoder instance"""
        self.config = config if config is not None else EmbeddingConfig()
        self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            use_fast=True,
        )

        if not self.tokenizer.is_fast:
            raise ValueError("Semantic segmentation requires a fast tokenizer")

        self.special_token_count = self.tokenizer.num_special_tokens_to_add(
            pair=False,
        )

        if self.config.max_tokens <= self.special_token_count:
            raise ValueError("Token budget leave no room for text")

        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            revision=self.config.revision,
            dtype=torch.float32,
        )

        torch_model = cast(torch.nn.Module, self.model)
        torch_model.to(device=self.device)
        self.model.eval()

        self.dimension = int(self.model.config.hidden_size)
        self.content_token_budget = (
            self.config.max_tokens - self.special_token_count
        )

    def segment_chunks(self, chunks: list[Chunk]) -> list[TokenSegment]:
        """Split chunks into bounded token sequence without overlap"""
        segments: list[TokenSegment] = []

        for chunk_position, chunk in enumerate(chunks):
            encoded = self.tokenizer(
                chunk.text,
                truncation=True,
                max_length=self.config.max_tokens,
                return_overflowing_tokens=True,
                stride=0,
                padding=False,
            )

            for segment_position, token_ids in enumerate(encoded["input_ids"]):
                segments.append(
                    TokenSegment(
                        chunk_position=chunk_position,
                        segment_position=segment_position,
                        input_ids=token_ids,
                    )
                )

        return segments

    def _encode_batch(self, input_ids: list[list[int]]) -> torch.Tensor:
        """Encode one nonempty batch of prepared token sequences."""
        inputs = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            token_embeddings = cast(
                torch.Tensor,
                self.model(**inputs).last_hidden_state,
            )

            mask = inputs["attention_mask"].unsqueeze(-1).to(
                dtype=token_embeddings.dtype
            )

            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            pooled = summed / counts

        return F.normalize(pooled, p=2, dim=1)

    def encode_segments(self, segments: list[TokenSegment]) -> torch.Tensor:
        """Encode segments in CPU batches while preseving their order."""
        if not segments:
            return torch.empty(
                (0, self.dimension),
                dtype=torch.float32,
                device=self.device,
            )
        if any(
            len(segment.input_ids) > self.config.max_tokens
            for segment in segments
        ):
            raise ValueError("A segment exceeds the token budget")

        batches: list[torch.Tensor] = []

        for start in tqdm(
            range(0, len(segments), self.config.batch_size),
            desc="Encoding segments",
            unit="batch",
        ):
            batch_segments = segments[
                start:start + self.config.batch_size
            ]

            batch_ids = [
                segment.input_ids
                for segment in batch_segments
            ]
            batches.append(self._encode_batch(batch_ids))
        return torch.cat(batches, dim=0)

    def encode_query(self, query: str) -> torch.Tensor:
        """Encode one nonblank question without silently truncating it"""
        if not query.strip():
            raise ValueError("Question must not be blank")

        encoded = self.tokenizer(
            query,
            truncation=False,
        )

        token_ids = cast(list[int], encoded["input_ids"])

        if len(token_ids) > self.config.max_tokens:
            raise ValueError("Question exceeds the token budget")

        return self._encode_batch([token_ids])
