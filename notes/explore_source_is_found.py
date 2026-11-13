"""Observe how source IoU becomes a found-or-missing decision"""

from src.evaluation import source_iou, source_is_found
from src.models import MinimalSource


def main() -> None:
    """Compare one reference with a growing list of results."""
    reference = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    candidates = [
        MinimalSource(
            file_path="data/raw/other.py",
            first_character_index=100,
            last_character_index=200,
        ),
        MinimalSource(
            file_path="data/raw/example.py",
            first_character_index=199,
            last_character_index=200,
        ),
        MinimalSource(
            file_path="data/raw/example.py",
            first_character_index=100,
            last_character_index=300,
        ),
    ]

    print("REFERENCE :", reference)
    print("\nSource trouvee sans resultats :", source_is_found(reference, []))

    for k in range(1, len(candidates) + 1):
        selected = candidates[:k]
        latest = selected[-1]

        print(f"\nTOP-{k}")
        print("Nouveau chunk :", latest)
        print("Son IoU :", source_iou(reference, latest))
        print("Source trouvee :", source_is_found(reference, selected))


if __name__ == "__main__":
    main()
