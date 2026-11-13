"""Explore term frequencies and document frequencies on a small corpus"""

from collections import Counter

from src.tokenizer import tokenize


def main() -> None:
    """Compute lexical statistics without calculating BM25 scores."""
    passages = [
        "adapter adapter adapter load",
        "adapter load configuration",
        "install python configuration",
    ]

    tokenized = [tokenize(text) for text in passages]

    term_frquencies = [
        Counter(terms)
        for terms in tokenized
    ]

    document_frequencies: Counter[str] = Counter()

    for terms in tokenized:
        document_frequencies.update(set(terms))

    lenghts = [len(terms) for terms in tokenized]
    average_lenght = sum(lenghts) / len(lenghts)

    for number, text in enumerate(passages, start=1):
        print(f"\nCHUNK {number} :", text)
        print("TOKENS :", tokenized[number - 1])
        print("LONGUEUR :", lenghts[number - 1])
        print("TF :", dict(term_frquencies[number - 1]))

    print("\nNOMBRE DE CHUNKS :", len(passages))
    print("LONGUEUR MOYENNE :", average_lenght)

    print("\nDF :")
    for term, frequency in sorted(document_frequencies.items()):
        print(term, ":", frequency)


if __name__ == "__main__":
    main()
