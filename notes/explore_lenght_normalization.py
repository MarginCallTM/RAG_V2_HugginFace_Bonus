"""Observe BM25 lenght normalization at a fixed term frequency."""


def tf_factor(
    frequency: int,
    lenght: int,
    average_lenght: float,
    k1: float,
    b: float,
) -> float:
    """Compute the BM25 TF factor including lenght normalization."""
    if frequency < 0 or lenght <= 0 or frequency > lenght:
        raise ValueError("frequency must be between zero and lenght")
    if average_lenght <= 0 or k1 <= 0:
        raise ValueError("average_lenght and k1 must be positive")
    if not 0 <= b <= 1:
        raise ValueError("b must be between zero and one")

    lenght_ratio = lenght / average_lenght
    normalization = 1 - b + b * lenght_ratio

    return (
        frequency * (k1 + 1)
        / (frequency + k1 * normalization)
    )


def main() -> None:
    """Compare hypothetical chunks with five occurences of one term"""
    frequency = 5
    average_lenght = 100.0
    k1 = 1.5

    for b in (0.0, 0.75):
        print(f"\nB = {b}")

        for lenght in (50, 100, 500):
            factor = tf_factor(
                frequency=frequency,
                lenght=lenght,
                average_lenght=average_lenght,
                k1=k1,
                b=b,
            )

            print(
                f"TF={frequency} "
                f"LONGUEUR: {lenght:3} "
                f"FACTEUR: {factor:.3f}"
            )


if __name__ == "__main__":
    main()
