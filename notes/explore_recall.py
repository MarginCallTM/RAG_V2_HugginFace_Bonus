"""Observe recall for one question with two reference sources."""

from src.evaluation import recall_at_k, source_is_found
from src.models import MinimalSource


def main() -> None:
    """Show the effect of ranking and repeated matching chunks."""
    source_a = MinimalSource(
        file_path="data/raw/a.py",
        first_character_index=100,
        last_character_index=200,
    )
    source_b = MinimalSource(
        file_path="data/raw/b.py",
        first_character_index=100,
        last_character_index=200,
    )

    references = [source_a, source_b]
    retrieved = [source_a, source_a, source_a, source_b]

    print("Une question attend deux sources : A et B.")
    print("Classement des chunks recuperes : A, A, A, B")

    for k in range(5):
        selected = retrieved[:k]

        print(f"\nTOP-{k}")
        for reference in references:
            found = source_is_found(reference, selected)
            print(f"  {reference.file_path} : retrouve = {found}")

        score = recall_at_k(references, retrieved, k=k)
        print(f"   Recall@{k} : {score:.0%}")


if __name__ == "__main__":
    main()
