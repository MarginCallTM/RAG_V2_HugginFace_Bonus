"""Reuse semantic segments during incremental index updates."""

from src.embeddings import TokenSegment


def remap_reusable_segments(
    previous_segments: list[TokenSegment],
    old_to_new: dict[int, int],
) -> tuple[list[TokenSegment], list[int]]:
    """Remap retianed segments and identify their previous vector rows."""
    current_segments: list[TokenSegment] = []
    retained_rows: list[int] = []

    for row, segment in enumerate(previous_segments):
        if segment.chunk_position not in old_to_new:
            continue

        current_segments.append(
            TokenSegment(
                chunk_position=old_to_new[segment.chunk_position],
                segment_position=segment.segment_position,
                input_ids=segment.input_ids,
            )
        )
        retained_rows.append(row)

    return current_segments, retained_rows
