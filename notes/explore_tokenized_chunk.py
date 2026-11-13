"""Compare query terms with terms from a real reference-region chunk."""

from collections import Counter
from pathlib import Path

from src.indexer import load_index
from src.models import AnsweredDataset
from src.tokenizer import tokenize


def main() -> None:
    """Inspect lexical matches without computing retrieval scores."""
    dataset_path = Path(
        "data/datasets/AnsweredQuestions/dataset_docs_public.json"
    )

    with dataset_path.open(encoding="utf-8") as handle:
        dataset = AnsweredDataset.model_validate_json(handle.read())

    question = dataset.rag_questions[0]
    source = question.sources[0]
    index = load_index(Path("data/processed"))

    candidates = [
        chunk
        for chunk in index.chunks
        if chunk.file_path == source.file_path
        and chunk.first_character_index < source.last_character_index
        and chunk.last_character_index > source.first_character_index
    ]

    if not candidates:
        raise ValueError("No chunk overlaps the reference source")

    chunk = candidates[0]
    original_text = chunk.text

    query_terms = tokenize(question.question)
    chunks_terms = tokenize(chunk.text)
    frequencies = Counter(chunks_terms)
    shared_terms = sorted(set(query_terms) & set(chunks_terms))

    print("QUESTION :", question.question)
    print("TERMES DE LA QUESTION :", query_terms)

    print("\nSOURCE :", chunk.file_path)
    print(
        "BORNES :",
        chunk.first_character_index,
        chunk.last_character_index,
    )
    print("LONGUEUR EN CARACTERES :", len(chunk.text))
    print("NOMBRE DE TOKEN LEXICAUX :", len(chunks_terms))
    print("EXTRAIT :", repr(chunk.text[:300]))

    print("\nPREMIERS TOKENS :", chunks_terms[:50])
    print("\nTERMES COMMUNS:", shared_terms)

    print("\nFREQUENCES DANS LE CHUNK :")
    for term in shared_terms:
        print(term, ":", frequencies[term])

    print("\nTEXTE ORIGINAL INCHANGE :", chunk.text == original_text)


if __name__ == "__main__":
    main()
