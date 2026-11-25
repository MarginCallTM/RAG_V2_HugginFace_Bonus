"""Explore vector reuse when chunk positions change."""

import torch

from src.embeddings import TokenSegment
from src.semantic_incremental import remap_reusable_segments


def main() -> None:
    """Remove one chunk and remap the remaining segment references."""
    previous_segments = [
        TokenSegment(
            chunk_position=0,
            segment_position=0,
            input_ids=[101, 200, 102],
        ),
        TokenSegment(
            chunk_position=0,
            segment_position=1,
            input_ids=[101, 201, 102],
        ),
        TokenSegment(
            chunk_position=1,
            segment_position=0,
            input_ids=[101, 300, 102],
        ),
        TokenSegment(
            chunk_position=1,
            segment_position=1,
            input_ids=[101, 301, 102],
        ),
        TokenSegment(
            chunk_position=2,
            segment_position=0,
            input_ids=[101, 400, 102],
        ),
    ]

    previous_vectors = torch.tensor([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    old_to_new = {
        1: 0,
        2: 1,
    }

    current_segments: list[TokenSegment] = []
    retained_rows: list[int] = []

    current_segments, retained_rows = remap_reusable_segments(
        previous_segments,
        old_to_new,
    )

    current_vectors = previous_vectors[retained_rows]

    print("Previous vectors:")
    print(previous_vectors)

    print("\nRetained old rows:", retained_rows)

    print("\nCurrent vectors:")
    print(current_vectors)

    for row, segment in enumerate(current_segments):
        print(
            f"New vector row {row} "
            f"-> new chunk {segment.chunk_position}, "
            f"segment {segment.segment_position}"
        )

    print(
        "\nVector values preserved:",
        torch.equal(current_vectors, previous_vectors[2:]),
    )


if __name__ == "__main__":
    main()
