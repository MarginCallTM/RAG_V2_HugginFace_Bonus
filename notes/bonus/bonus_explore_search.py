"""Search previously saved passage embeddings without re-encoding them."""

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from notes.bonus.bonus_explore_encoder import encode_texts


def main() -> None:
    """Load passage vectors and encode only the questions."""
    index_path = Path(
        "data/output/experiments/semantic/persistence/demo_index.pt"
    )

    if not index_path.is_file():
        print("Index missing: run bonus_explore_encoder first")
        return

    print("Loading saved passage vectors...", flush=True)
    artifact = torch.load(
        index_path,
        map_location="cpu",
        weights_only=True,
    )

    passages = artifact["passages"]
    passage_embeddings = artifact["embeddings"]
    model_name = artifact["model_name"]

    print("Loaded passages:", len(passages))
    print("Loaded vectors shape:", tuple(passage_embeddings.shape))

    print("Loading MiniLM for quesiton encoding...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to("cpu")
    model.eval()

    questions = [
        "How can I reduce GPU memory usage?",
        "How do I change the port used by the web server?",
    ]

    print("Encoding questions only...", flush=True)
    question_embeddings = encode_texts(questions, tokenizer, model)

    scores = question_embeddings @ passage_embeddings.T
    k = 2

    for question_index, question in enumerate(questions):
        print(f"\nQuestion: {question}")

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
                f"{rank}. cosine={score:.4f} | "
                f"{passages[passage_index]}"
            )


if __name__ == "__main__":
    main()
