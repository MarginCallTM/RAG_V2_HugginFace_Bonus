"""Split source text into chunks with character offsets."""

import re
import ast

from src.models import Chunk
from pathlib import Path


def line_starts(text: str) -> list[int]:
    """Return the character offset at the start of each source line"""
    starts = [0]

    for index, character in enumerate(text):
        if character == "\n" and index + 1 < len(text):
            starts.append(index + 1)
    return starts


def statement_start(
    node: ast.stmt,
    starts: list[int],
    text: str,
) -> int:
    """Locate a statement with decorators and adjacent leading comments."""
    first_line = node.lineno

    if isinstance(
        node,
        (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
    ):
        for decorator in node.decorator_list:
            first_line = min(first_line, decorator.lineno)

    line_index = first_line - 1

    while line_index > 0:
        previous_line = text[
            starts[line_index - 1]: starts[line_index]
        ]

        if not previous_line.lstrip().startswith("#"):
            break

        line_index -= 1

    return starts[line_index]


def statement_spans(
    text: str,
    statements: list[ast.stmt],
    span: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Partition a source range at statement starts."""
    if span is None:
        span = (0, len(text))

    first, last = span

    if not 0 <= first <= last <= len(text):
        raise ValueError("span must be within the source text")

    if first == last:
        return []

    starts = line_starts(text)
    boundaries = {first}

    for node in statements:
        position = statement_start(node, starts, text)

        if first < position < last:
            boundaries.add(position)
    ordered = sorted(boundaries)
    ordered.append(last)

    return [
        (start, end)
        for start, end in zip(ordered, ordered[1:])
    ]


def split_fixed(
    file_path: str,
    text: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """Split text into consecutive chunks of bounded size.

    Args:
        file_path: Source path stored in every chunk.
        text: Original text to split.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        Chunks in source order, without gaps or overlap.

    Raises:
        ValueError: If max_chunk_size is not positive.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    chunks: list[Chunk] = []

    for start in range(0, len(text), max_chunk_size):
        end = min(start + max_chunk_size, len(text))

        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end,
                text=text[start:end]
            )
        )
    return chunks


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Locate paragraph blocks while preserving all characters.

    Each block includes the blank-line separator following it.
    Returned ranges cover the text without gaps or overlap.
    """

    spans: list[tuple[int, int]] = []
    start = 0

    for separator in re.finditer(r"\n(?:[ \t]*\n)+", text):
        end = separator.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def split_text(
    file_path: str,
    text: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """Pack paragraphs blocks into chunks of bounded size.

    Oversized paragraphs are split using fixed-size boundaries.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    chunks: list[Chunk] = []
    current_start = 0
    current_end = 0

    for start, end in paragraph_spans(text):
        if end - current_start <= max_chunk_size:
            current_end = end
            continue

        if current_end > current_start:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=current_start,
                    last_character_index=current_end,
                    text=text[current_start:current_end],
                )
            )

        if end - start > max_chunk_size:
            pieces = split_fixed(
                file_path,
                text[start:end],
                max_chunk_size,
            )

            for piece in pieces:
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        first_character_index=(
                            start + piece.first_character_index
                        ),
                        last_character_index=(
                            start + piece.last_character_index
                        ),
                        text=piece.text,
                    )
                )
            current_start = end
            current_end = end
        else:
            current_start = start
            current_end = end
    if current_end > current_start:
        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=current_start,
                last_character_index=current_end,
                text=text[current_start:current_end]
            )
        )
    return chunks


def markdown_sections(text: str) -> list[tuple[int, int]]:
    """Locate Markdown sections while ignoring fenced code content."""
    boundaries = [0]
    offset = 0
    fence_marker = ""
    fence_lenght = 0

    for line in text.splitlines(keepends=True):
        fence = re.match(r" {0,3}(`{3,}|~{3,})(.*)", line)

        if fence_marker:
            if fence:
                marker = fence.group(1)
                remainder = fence.group(2)

                if (
                    marker[0] == fence_marker
                    and len(marker) >= fence_lenght
                    and not remainder.strip()
                ):
                    fence_marker = ""
                    fence_lenght = 0

        elif fence:
            marker = fence.group(1)
            fence_marker = marker[0]
            fence_lenght = len(marker)

        elif re.match(r"#{1,6}[ \t]+", line) and offset > 0:
            boundaries.append(offset)

        offset += len(line)

    if boundaries[-1] != len(text):
        boundaries.append(len(text))

    return [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:])
    ]


def split_markdown(
    file_path: str,
    text: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """SPlit Markdown sections into chunks with document offsets"""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    chunks: list[Chunk] = []

    for section_start, section_end in markdown_sections(text):
        section_text = text[section_start:section_end]

        pieces = split_text(
            file_path,
            section_text,
            max_chunk_size
        )

        for piece in pieces:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=(
                        section_start + piece.first_character_index
                    ),
                    last_character_index=(
                        section_start + piece.last_character_index
                    ),
                    text=piece.text,
                )
            )

    return chunks


def python_spans(
    text: str,
    statements: list[ast.stmt],
    span: tuple[int, int],
    max_chunk_size: int,
) -> list[tuple[int, int]]:
    """Recursively subdivide oversized Python statement groups"""
    blocks = statement_spans(text, statements, span)
    groups = pack_spans(blocks, max_chunk_size)
    starts = line_starts(text)
    result: list[tuple[int, int]] = []

    for start, end in groups:
        if end - start <= max_chunk_size:
            result.append((start, end))
            continue

        children: list[ast.stmt] = []

        for node in statements:
            position = statement_start(node, starts, text)

            if start <= position < end:
                children.extend(child_statements(node))

        inner_blocks = statement_spans(
            text,
            children,
            span=(start, end),
        )

        if children and len(inner_blocks) > 1:
            result.extend(
                python_spans(
                    text,
                    children,
                    (start, end),
                    max_chunk_size,
                )
            )
        else:
            pieces = split_text(
                "temporary.py",
                text[start:end],
                max_chunk_size,
            )

            for piece in pieces:
                result.append(
                    (
                        start + piece.first_character_index,
                        start + piece.last_character_index,
                    )
                )
    return result


def split_python(
    file_path: str,
    text: str,
    max_chunk_size: int,
) -> list[Chunk]:
    """Split python using recursive synthax boundaries and text fallback"""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return split_text(file_path, text, max_chunk_size)

    spans = python_spans(
        text,
        tree.body,
        (0, len(text)),
        max_chunk_size,
    )

    return [
        Chunk(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end,
            text=text[start:end],
        )
        for start, end in spans
    ]


def pack_spans(
    spans: list[tuple[int, int]],
    max_chunk_size: int,
) -> list[tuple[int, int]]:
    """Groupe contiguous spans without splitting oversized blocks"""
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    if not spans:
        return []

    packed: list[tuple[int, int]] = []
    current_start, current_end = spans[0]

    for start, end in spans[1:]:
        if end - current_start <= max_chunk_size:
            current_end = end
        else:
            packed.append((current_start, current_end))
            current_start, current_end = start, end
    packed.append((current_start, current_end))

    return packed


def child_statements(node: ast.AST) -> list[ast.stmt]:
    """Return statements directly contained in an AST node"""
    return [
        child
        for child in ast.iter_child_nodes(node)
        if isinstance(child, ast.stmt)
    ]


def chunk_file(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000,
) -> list[Chunk]:
    """Select the chunking strategy from the source file extension."""
    if not 1 <= max_chunk_size <= 2000:
        raise ValueError("max_chunk_size must be between 1 and 2000")

    extension = Path(file_path).suffix

    if extension == ".py":
        return split_python(file_path, text, max_chunk_size)
    if extension == ".md":
        return split_markdown(file_path, text, max_chunk_size)
    if extension == ".txt":
        return split_text(file_path, text, max_chunk_size)
    raise ValueError(f"Unsupported file extension: {extension}")
