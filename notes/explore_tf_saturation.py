"""Observe BM25 term-frequency saturation at average document lenght"""


def tf_factor(frequency: int, k1: float) -> float:
    """Compute the TF factor for a chunk of average length."""
    if frequency < 0:
        raise ValueError("frequency must be non-negative")

    if k1 <= 0:
        raise ValueError("k1 must be positive")

    return frequency * (k1 + 1) / (frequency + k1)


def main() -> None:
    """Compare raw frequency with a saturating TF factor."""
    k1 = 1.5

    print("K1 :", k1)

    for frequency in (0, 1, 2, 3, 10, 100):
        factor = tf_factor(frequency, k1)

        print(
            f"TF={frequency:3} "
            f"FACTEUR={factor:.3f}"
        )


if __name__ == "__main__":
    main()
