"""Explore reusable CPU encoding for passage and questions."""

import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase
)
from tqdm import tqdm


def encode_texts(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
) -> torch.Tensor:
    """Encode a nonempty list of texts into normlaized CPU vectors."""
    if not texts:
        raise ValueError("Provide as least one text")

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.inference_mode():
        token_embeddings = model(**encoded).last_hidden_state

        mask = encoded["attention_mask"].unsqueeze(-1).to(
            dtype=token_embeddings.dtype
        )

        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts

        return F.normalize(pooled, p=2, dim=1)


def encode_in_batches(
    texts: list[str],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    batch_size: int = 2,
) -> torch.Tensor:
    """Encode texts in bounded batches while preserving their order"""
    if not texts:
        raise ValueError("Provide at least one text")
    if batch_size <= 0:
        raise ValueError("batch_size must be postive")

    batches: list[torch.Tensor] = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding",
        unit="batch",
    ):
        batch_texts = texts[start:start + batch_size]
        embeddings = encode_texts(batch_texts, tokenizer, model)
        batches.append(embeddings)

    return torch.cat(batches, dim=0)


def main() -> None:
    """Encode passage on CPU and save their vectors and metadata."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("Loading MiniLM on CPU...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to("cpu")
    model.eval()

    passages = [
        "Eat your pasta every day.",
        "Decrease the amount of device RAM allocated to the cache.",
        "Configure the HTTP server port.",
    ]

    print("Encoding passages...", flush=True)
    passage_embeddings = encode_in_batches(
        passages,
        tokenizer,
        model,
        batch_size=2,
    )

    output_dir = Path("data/output/experiments/semantic/persistence")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "demo_index.pt"

    artifact = {
        "format_version": 1,
        "model_name": model_name,
        "max_length": 256,
        "normalized": True,
        "passages": passages,
        "embeddings": passage_embeddings,
    }

    torch.save(artifact, index_path)
    print("Saved index:", index_path)


if __name__ == "__main__":
    main()
