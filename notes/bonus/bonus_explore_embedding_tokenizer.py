"""Inspect the tokenizer used by the MiniLM embedding model."""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def main() -> None:
    """Display tokens, vocabulary IDs and input tensor dimensions."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    texts = [
        "How can I reduce GPU memory usage?",
        "Eat your pasta every day",
        "Download the last world of warcrafts",
        "Decrease the amount of device RAM allocated to the cache.",
        "Configure the HTTP server port.",
        "The cat is sleeping on the sofa.",
        "Artificial intelligence is changing the world.",
        "I forgot my keys at home.",
        "The weather is beautiful today.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoded = tokenizer(
        texts,
        padding=True,
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )

    print("Loading MiniLM on CPU...", flush=True)

    model = AutoModel.from_pretrained(model_name)
    model.to("cpu")
    model.eval()

    print("MiniLM loaded on CPU", flush=True)
    print("Computing token embeddings...", flush=True)

    with torch.inference_mode():
        outputs = model(**encoded)

    print("Token embeddings computed.", flush=True)
    token_embeddings = outputs.last_hidden_state

    attention_mask = encoded["attention_mask"]

    pooling_mask = attention_mask.unsqueeze(-1).to(
        dtype=token_embeddings.dtype
    )

    masked_embeddings = token_embeddings * pooling_mask
    summed_embeddings = masked_embeddings.sum(dim=1)

    valid_token_counts = pooling_mask.sum(dim=1).clamp(min=1.0)
    sentence_embeddings = summed_embeddings / valid_token_counts

    normalized_embeddings = F.normalize(
        sentence_embeddings,
        p=2,
        dim=1,
    )

    norms_before = sentence_embeddings.norm(p=2, dim=1)
    norms_after = normalized_embeddings.norm(p=2, dim=1)

    print("Norms before normalization:", norms_before.tolist())
    print("Norms after normalization:", norms_after.tolist())
    print("Normalized embeddings shape:", tuple(normalized_embeddings.shape))

    query_embedding = normalized_embeddings[0]

    print("\nSEMANTIC SEARCH")
    print("Question:", texts[0])

    results: list[tuple[(int, float)]] = []

    for index in range(1, len(texts)):
        score = torch.dot(
            query_embedding,
            normalized_embeddings[index],
        ).item()

        results.append((index, score))

    ranked_results = sorted(
        results,
        key=lambda result: result[1],
        reverse=True,
    )

    k = 3

    for rank, (index, score) in enumerate(ranked_results[:k], start=1):
        print(f"{rank}. cosine={score:.4f} | {texts[index]}")

    print("\nPooling mask shape:", tuple(pooling_mask.shape))
    print("Summed embeddings shape:", tuple(summed_embeddings.shape))
    print("Valid token counts:", valid_token_counts.squeeze(-1).tolist())
    print("Sentence embeddings shape:", tuple(sentence_embeddings.shape))
    print("First text, first coordinates", sentence_embeddings[0, :5].tolist())

    print("Token embeddings shape:", tuple(token_embeddings.shape))
    print("First position shape:", tuple(token_embeddings[0, 0].shape))
    print("First five coordinates:", token_embeddings[0, 0, :5].tolist())
    print("Output device:", token_embeddings.device)

    print("Input shape:", tuple(encoded["input_ids"].shape))
    print("Attention mask:", encoded["attention_mask"].tolist())

    for index, text in enumerate(texts):
        token_ids = encoded["input_ids"][index].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        mask = encoded["attention_mask"][index].tolist()

        print(f"\nText {index + 1}: {text}")
        print(f"{'Position':>8} {'Token':>15} {'ID':>8} {'Mask':>6}")

        for position, (token, token_id, active) in enumerate(
            zip(tokens, token_ids, mask)
        ):
            print(
                f"{position:>8} {token:>15} "
                f"{token_id:>8} {active:>6}"
            )

        print("Positions prises en compte:", sum(mask))
        print("Positions de padding:", len(mask) - sum(mask))


if __name__ == "__main__":
    main()
