"""Observe token segmentation without losing the end of a passage."""

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from notes.bonus.bonus_explore_encoder import encode_texts


def main() -> None:
    """Split long passages into bounded token sequences."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    max_length = 256
    special_count = tokenizer.num_special_tokens_to_add(pair=False)

    print("Maximum positions per segment:", max_length)
    print("Special tokens per segment:", special_count)
    print("Content capacity:", max_length - special_count)

    prefix = "cache " * 280
    passages = [
        prefix + "The HTTP server listens on port 8000.",
        prefix + "The HTTP server listens on port 9000.",
    ]

    all_segment_ids: list[list[int]] = []
    segment_origins: list[tuple[int, int]] = []

    for passage_index, passage in enumerate(passages):
        encoded = tokenizer(
            passage,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            stride=0,
            padding=False,
        )

        segments = encoded["input_ids"]

        print(f"\nPassage: {passage_index}")
        print("Characters:", len(passage))
        print("Number of segments:", len(segments))

        for segment_index, token_ids in enumerate(segments):
            all_segment_ids.append(token_ids)
            segment_origins.append((passage_index, segment_index))
            print(f"  Segment {segment_index}")
            print("  Token count:", len(token_ids))
            print(
                "  Within budget:",
                len(token_ids) <= max_length,
            )
            print(
                "  Ending:",
                tokenizer.decode(
                    token_ids[-20:],
                    skip_special_tokens=True,
                ),
            )

    print("\nPreparing the segment batch...", flush=True)

    segment_inputs = tokenizer.pad(
        {"input_ids": all_segment_ids},
        padding=True,
        return_tensors="pt",
    )

    print("Segment input shape:", tuple(segment_inputs["input_ids"].shape))
    print("Segment origins:", segment_origins)

    print("Loading MiniLM on CPU...", flush=True)
    model = AutoModel.from_pretrained(model_name)
    model.to("cpu")
    model.eval()

    print("Encoding segments...", flush=True)

    with torch.inference_mode():
        token_embeddings = model(**segment_inputs).last_hidden_state

        mask = segment_inputs["attention_mask"].unsqueeze(-1).to(
            dtype=token_embeddings.dtype
        )

        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts

        segment_embeddings = F.normalize(pooled, p=2, dim=1)

    print("Segment embeddings shape:", tuple(segment_embeddings.shape))
    print("Segment norms", segment_embeddings.norm(dim=1).tolist())

    for row, (passage_index, segment_index) in enumerate(segment_origins):
        print(
            f"Vector {row} -> passage {passage_index}, "
            f"segment {segment_index}"
        )

    question = "Which port does the HTTP server listen on?"

    print("\nEncoding the question...", flush=True)
    question_embedding = encode_texts([question], tokenizer, model)

    scores = question_embedding @ segment_embeddings.T

    print("Question:", question)
    print("Segment scores shape:", tuple(scores.shape))

    best_by_passage: dict[int, tuple[int, float]] = {}

    for row, score in enumerate(scores[0].tolist()):
        passage_index, segment_index = segment_origins[row]

        print(
            f"Passage {passage_index}, segment {segment_index}: "
            f"cosine={score:.4f}"
        )

        if passage_index not in best_by_passage:
            best_by_passage[passage_index] = (segment_index, score)
        elif score > best_by_passage[passage_index][1]:
            best_by_passage[passage_index] = (segment_index, score)

    ranked_passages = sorted(
        best_by_passage.items(),
        key=lambda item: item[1][1],
        reverse=True,
    )

    print("\nPASSAGE RANKING")

    for rank, (passage_index, (segment_index, score)) in enumerate(
        ranked_passages,
        start=1,
    ):
        print(
            f"{rank}. Passage {passage_index}: "
            f"cosine={score:.4f}, best segment={segment_index}"
        )
        print("  Original ending:", passages[passage_index][-80:])


if __name__ == "__main__":
    main()
