"""Observe dot products and cosine similarity on artificial vectors."""

from math import sqrt


def dot(left: list[float], right: list[float]) -> float:
    """Compute the dot product of equal-sized vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")

    return sum(a * b for a, b in zip(left, right))


def norm(vector: list[float]) -> float:
    """Compute the Euclidean length of a vector."""
    return sqrt(dot(vector, vector))


def cosine(left: list[float], right: list[float]) -> float:
    """Compare vector directions independently of their lengths."""
    denominator = norm(left) * norm(right)

    if denominator == 0:
        raise ValueError("Cosine is undefined for a zero vector")

    return dot(left, right) / denominator


def main() -> None:
    """Compare a query vector with three artificial candidates."""
    query = [1.0, 0.0]

    candidates = {
        "A": [0.8, 0.2],
        "B": [0.0, 1.0],
        "A multiplied by 10": [8.0, 2.0],
    }

    for name, vector in candidates.items():
        print(
            f"{name}: "
            f"norm={norm(vector):.3f}, "
            f"dot={dot(query, vector):.3f}, "
            f"cosine={cosine(query, vector):.3f}"
        )


if __name__ == "__main__":
    main()
