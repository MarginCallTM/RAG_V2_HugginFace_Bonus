"""Encode a question using the saved semantic index configuration"""

from pathlib import Path

from src.embeddings import SemanticEncoder
from src.semantic_index import load_semantic_index


def main() -> None:
    """Load the index settings and inspect a normalized query vector"""
    path = Path(
        "data/output/experiments/semantic/sample/semantic_index.pt",
    )
    data, _ = load_semantic_index(path)

    print("Loading encoder from index configuration...", flush=True)
    encoder = SemanticEncoder(data.metadata.encoder_config)

    question = "How can I reduce GPU memory usage?"
    embedding = encoder.encode_query(question)

    print("Question:", question)
    print("Query shape:", tuple(embedding.shape))
    print("Query norm:", embedding.norm(dim=1).tolist())
    print("Device:", embedding.device)


if __name__ == "__main__":
    main()
