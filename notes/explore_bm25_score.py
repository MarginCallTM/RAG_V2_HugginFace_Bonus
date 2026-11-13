"""Explain a complete BM-25-style score a small corpus"""

from collections import Counter
from math import log1p

from src.tokenizer import tokenize


def main() -> None:
    """Display term contributions and rank three passages."""
    passages = [
        "Load a LoRA adapter using a POST request.",
        "Install Python before starting the server.",
        "The adapter configuration defines the adapter name.",
    ]
    question = "load adapter"

    k1 = 1.5
    b = 0.75

    tokenized = [tokenize(text) for text in passages]
    frequencies = [Counter(terms) for terms in tokenized]
    lengths = [len(terms) for terms in tokenized]

    total_chunks = len(passages)
    average_length = sum(lengths) / total_chunks

    document_frequencies: Counter[str] = Counter()
    for terms in tokenized:
        document_frequencies.update(set(terms))

    query_terms = sorted(set(tokenize(question)))
    results: list[tuple[int, float]] = []

    print("QUESTION :", question)
    print("TERMES RECHERCHÉS :", query_terms)
    print("LONGUEUR MOYENNE :", average_length)

    for index, passage in enumerate(passages):
        score = 0.0
        length_ratio = lengths[index] / average_length
        normalization = 1 - b + b * length_ratio

        print(f"\nCHUNK {index + 1} :", passage)
        print("LONGUEUR :", lengths[index])

        for term in query_terms:
            tf = frequencies[index][term]
            df = document_frequencies[term]

            if df == 0:
                print(term, ": absent du corpus")
                continue

            idf = log1p(
                (total_chunks - df + 0.5) / (df + 0.5)
            )

            factor = (
                tf * (k1 + 1)
                / (tf + k1 * normalization)
            )

            contribution = idf * factor
            score += contribution

            print(
                f"  {term}: TF={tf}, DF={df}, "
                f"IDF={idf:.3f}, FACTEUR={factor:.3f}, "
                f"CONTRIBUTION={contribution:.3f}"
            )

        results.append((index, score))
        print(f"SCORE TOTAL : {score:.3f}")

    ranked = sorted(
        results,
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nCLASSEMENT :")
    for rank, (index, score) in enumerate(ranked, start=1):
        print(f"{rank}. CHUNK {index + 1} — score {score:.3f}")


if __name__ == "__main__":
    main()
