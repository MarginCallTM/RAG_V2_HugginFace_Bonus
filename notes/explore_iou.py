"""Observe source IoU on small character ranges without loading the corpus."""

from src.evaluation import source_iou
from src.models import MinimalSource


def main() -> None:
    """Compare one reference with several retrieved source examples."""
    reference = MinimalSource(
        file_path="data/raw/example.py",
        first_character_index=100,
        last_character_index=200,
    )
    examples = [
        ("Bornes identiques", reference.file_path, 100, 200),
        ("Chunk deux fois plus long", reference.file_path, 100, 300),
        ("Chevauchement partiel", reference.file_path, 150, 250),
        ("Chunk inclus dans la reference", reference.file_path, 120, 180),
        ("Plages qui se touchent", reference.file_path, 200, 300),
        ("Plages separees", reference.file_path, 300, 400),
        ("Autre fichier, memes bornes", "data/raw/other.py", 100, 200),
    ]

    print("REFERENCE :", reference.file_path, "[100:200]")
    print("Les fins sont exclues : [100:200] contient 100 caracteres.")

    for title, file_path, start, end in examples:
        retrieved = MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
        )
        print(f"\n{title}")
        print(f"  Chunk : {file_path} [{start}:{end}]")

        if reference.file_path == retrieved.file_path:
            # Detail pedagogique du calcul effectue par source_iou.
            intersection = max(0, min(200, end) - max(100, start))
            union = 100 + (end - start) - intersection
            print(f"  Intersection : {intersection} caracteres")
            print(f"  Union : {union} caracteres")
            print(f"  Calcul : {intersection} / {union}")
        else:
            print("  Chemins differents : aucune correspondance possible.")

        score = source_iou(reference, retrieved)
        print(f"  IoU retournee par source_iou : {score:.3f}")

    print("\nCe script observe l'IoU ; il ne calcule pas le recall.")
    print("Pipeline : sources -> IoU -> correspondance -> recall par question")
    print("           -> moyenne sur le dataset.")


if __name__ == "__main__":
    main()
