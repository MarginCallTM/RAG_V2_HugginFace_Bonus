"""Observe a positive IDF variant across document frequencies."""

from math import log1p


def compute_idf(total_chunks: int, document_frequency: int) -> float:
    """Compute a smoothed positive IDF for a term present in the corpus"""
    if total_chunks <= 0:
        raise ValueError("total_chunks must be positive")
    if not 1 <= document_frequency <= total_chunks:
        raise ValueError("document_frequency must be between 1 and N")

    ratio = (
        (total_chunks - document_frequency + 0.5)
        / (document_frequency + 0.5)
    )

    return log1p(ratio)


def main() -> None:
    """Compare the weight of common and rare terms"""
    total_chunks = 1000

    examples = [
        ("the", 900),
        ("adapter", 100),
        ("load_lora_adapter", 4),
    ]

    print("NOMBRE TOTAL DE CHUNKS :", total_chunks)

    for term, frequency in examples:
        weight = compute_idf(total_chunks, frequency)

        print(
            f"{term:20} "
            f"DF={frequency:4}"
            f"IDF={weight:.3f}"
        )


if __name__ == "__main__":
    main()
