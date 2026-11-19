"""Explore reusable CPU encoding for passage and questions."""

import torch
import torch.nn.functional as F
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
    """Encode passage and a question separately with one loaded model."""
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
    questions = [
        "How can I reduce GPU memory usage?",
        "How do I change the port used by the web server?",
    ]

    print("Encoding passages...", flush=True)
    passage_embeddings = encode_in_batches(
        passages,
        tokenizer,
        model,
        batch_size=2,
    )

    print("Encoding the question...", flush=True)
    question_embeddings = encode_texts(questions, tokenizer, model)

    print("Passages shape:", tuple(passage_embeddings.shape))
    print("Questions shape:", tuple(question_embeddings.shape))

    scores = question_embeddings @ passage_embeddings.T

    print("Score shape:", tuple(scores.shape))
    print("Scores:", scores.tolist())

    k = 2

    for question_index, question in enumerate(questions):
        print(f"\nQuestion {question_index}: {question}")

        ranked_results = sorted(
            enumerate(scores[question_index].tolist()),
            key=lambda result: result[1],
            reverse=True,
        )

        for rank, (passage_index, score) in enumerate(
            ranked_results[:k],
            start=1,
        ):
            print(
                f"{rank}. Passage {passage_index}: "
                f"cosine={score:.4f} | {passages[passage_index]}"
            )


if __name__ == "__main__":
    main()
