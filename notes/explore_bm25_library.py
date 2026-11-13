"""Observe BM250kapi scores on a small tokenized corpus"""

from rank_bm25 import BM25Okapi

from src.tokenizer import tokenize


def main() -> None:
    """Build BM25 statistics and rank passages for one question."""
    passages = [
        "Load a LoRA adapter using a POST request.",
        "Install Python before starting the server.",
        "The adapter configuration defines the adapter name.",
    ]
    question = "load adapter"

    tokenized = [tokenize(text) for text in passages]

    bm25 = BM25Okapi(
        tokenized,
        k1=1.5,
        b=0.75,
    )

    query_terms = sorted(set(tokenize(question)))
    scores = bm25.get_scores(query_terms)

    print("QUESTION :", question)
    print("TERMES :", query_terms)
    print("LONGUEUR MOYENNE :", bm25.avgdl)

    print("\nIDF DE LA LIB :")
    for term in query_terms:
        print(term, ":", bm25.idf.get(term, 0.0))

    ranked = sorted(
        enumerate(scores),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nCLASSEMENT :")
    for rank, (index, score) in enumerate(ranked, start=1):
        print(f"\n{rank}. CHUNK {index + 1} - score {score:.3f}")
        print(passages[index])


if __name__ == "__main__":
    main()
