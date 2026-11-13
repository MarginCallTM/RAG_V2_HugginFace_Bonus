"""Tests for text chunking and source offsets."""

import pytest
import ast

from src.chunking import (
    markdown_sections,
    paragraph_spans,
    split_fixed,
    split_text,
    split_markdown,
    line_starts,
    statement_start,
    statement_spans,
    split_python,
    pack_spans,
    child_statements,
    chunk_file,
)


def test_fixed_chunks_preserve_text_and_offsets() -> None:
    """Cover the original text exactly with bounded chunks."""
    original = "Bonjour. Voici un document."

    chunks = split_fixed("example.txt", original, max_chunk_size=10)

    assert [
        (chunk.first_character_index, chunk.last_character_index)
        for chunk in chunks
    ] == [(0, 10), (10, 20), (20, 27)]

    assert "".join(chunk.text for chunk in chunks) == original

    for chunk in chunks:
        start = chunk.first_character_index
        end = chunk.last_character_index

        assert chunk.file_path == "example.txt"
        assert chunk.text == original[start:end]
        assert 0 < len(chunk.text) <= 10


def test_fixed_chunks_handle_empty_text() -> None:
    """Return no chunks for an empty document."""
    assert split_fixed("empty.txt", "", max_chunk_size=10) == []


@pytest.mark.parametrize("size", [0, -1])
def test_fixed_chunks_reject_invalid_size(size: int) -> None:
    """Reject sizes that cannot produce non-empty chunks."""
    with pytest.raises(ValueError):
        split_fixed("example.txt", "Hello", max_chunk_size=size)


def test_paragraph_spans_preserve_separators() -> None:
    """Find blank lines without splitting on single newlines."""
    text = "ABC\nDEF\n\nGHI"

    assert paragraph_spans(text) == [(0, 9), (9, 12)]


def test_split_text_packs_neighboring_paragraphs() -> None:
    """Group paragraphs while respecting the size limit."""
    text = "ABC\n\nDEF\n\nGHI"

    chunks = split_text("example.txt", text, max_chunk_size=10)

    assert [
        (chunk.first_character_index, chunk.last_character_index)
        for chunk in chunks
    ] == [(0, 10), (10, 13)]

    assert [chunk.text for chunk in chunks] == [
        "ABC\n\nDEF\n\n",
        "GHI",
    ]


def test_split_text_preserves_offsets_after_oversized_paragraph() -> None:
    """Keep document offsets when subdividing a later paragraph."""
    text = "A\n\nBCDEFGHI"

    chunks = split_text("example.txt", text, max_chunk_size=5)

    assert [
        (chunk.first_character_index, chunk.last_character_index)
        for chunk in chunks
    ] == [(0, 3), (3, 8), (8, 11)]

    assert "".join(chunk.text for chunk in chunks) == text

    for chunk in chunks:
        start = chunk.first_character_index
        end = chunk.last_character_index

        assert chunk.text == text[start:end]
        assert chunk.file_path == "example.txt"
        assert 0 < len(chunk.text) <= 5


def test_split_text_handles_empty_document() -> None:
    """Return no chunks for empty text."""
    assert split_text("empty.txt", "", max_chunk_size=5) == []


@pytest.mark.parametrize("size", [0, -1])
def test_split_text_rejects_invalid_size(size: int) -> None:
    """Reject non-positive chunk sizes."""
    with pytest.raises(ValueError):
        split_text("example.txt", "ABC", max_chunk_size=size)


def test_markdown_sections_keep_headings_with_their_content() -> None:
    """Start each section at its heading."""
    text = (
        "# LoRA\n"
        "Présentation.\n"
        "## Charger\n"
        "Envoyer POST.\n"
        "## Retirer\n"
        "Envoyer DELETE.\n"
    )

    spans = markdown_sections(text)

    assert spans == [(0, 21), (21, 46), (46, 73)]

    assert [text[start:end] for start, end in spans] == [
        "# LoRA\nPrésentation.\n",
        "## Charger\nEnvoyer POST.\n",
        "## Retirer\nEnvoyer DELETE.\n",
    ]


def test_markdown_sections_preserve_intro_before_first_heading() -> None:
    """Keep text preceding the first heading."""
    text = "Introduction.\n\n## Charger\nInstructions."

    spans = markdown_sections(text)

    assert [text[start:end] for start, end in spans] == [
        "Introduction.\n\n",
        "## Charger\nInstructions.",
    ]


def test_markdown_sections_ignore_hash_without_space() -> None:
    """Require whitespace after the heading marker."""
    text = "Texte.\n#PasUnTitre\nSuite."

    assert markdown_sections(text) == [(0, len(text))]


def test_markdown_sections_handle_empty_document() -> None:
    """Return no sections for an empty document."""
    assert markdown_sections("") == []


def test_markdown_sections_ignore_headings_inside_code_fence() -> None:
    """Keep code comments inside their Markdown section."""
    text = (
        "# Guide\n"
        "```python\n"
        "# Commentaire Python\n"
        "print('Hello')\n"
        "```\n"
        "## Suite\n"
        "Instructions.\n"
    )

    spans = markdown_sections(text)

    assert len(spans) == 2
    assert text[spans[0][0]:spans[0][1]] == (
        "# Guide\n"
        "```python\n"
        "# Commentaire Python\n"
        "print('Hello')\n"
        "```\n"
    )
    assert text[spans[1][0]:spans[1][1]] == (
        "## Suite\nInstructions.\n"
    )


@pytest.mark.parametrize(
    ("opening", "inside"),
    [
        ("~~~python", "~~~"),
        ("````python", "```\n# Toujours dans le code\n````"),
        ("```python", "~~~\n# Toujours dans le code\n```"),
    ],
)
def test_markdown_sections_respect_fence_closure(
    opening: str,
    inside: str,
) -> None:
    """Close fences only with a compatible marker and length."""
    first_section = (
        "# Guide\n"
        f"{opening}\n"
        "# Commentaire\n"
        f"{inside}\n"
    )
    second_section = "## Suite\nInstructions.\n"
    text = first_section + second_section

    spans = markdown_sections(text)

    assert [text[start:end] for start, end in spans] == [
        first_section,
        second_section,
    ]


def test_markdown_sections_keep_unclosed_fence_open() -> None:
    """Treat the remaining document as code when no fence closes."""
    text = "# Guide\n```python\n# Commentaire\n## Encore du code\n"

    assert markdown_sections(text) == [(0, len(text))]


def test_split_markdown_keeps_sections_separate() -> None:
    """Keep distinct sections apart even when they could fit together."""
    text = "# A\nTexte.\n## B\nSuite."

    chunks = split_markdown("example.md", text, max_chunk_size=100)

    assert [chunk.text for chunk in chunks] == [
        "# A\nTexte.\n",
        "## B\nSuite.",
    ]


def test_split_markdown_preserves_document_offsets() -> None:
    """Preserve text and absolute offsets after section subdivision."""
    text = "# A\nCourt.\n## B\nUn paragraphe assez long."

    chunks = split_markdown("example.md", text, max_chunk_size=10)

    assert "".join(chunk.text for chunk in chunks) == text

    for chunk in chunks:
        start = chunk.first_character_index
        end = chunk.last_character_index

        assert chunk.file_path == "example.md"
        assert chunk.text == text[start:end]
        assert 0 < len(chunk.text) <= 10


def test_split_markdown_handles_empty_document() -> None:
    """Return no chunks for an empty Markdown document."""
    assert split_markdown("empty.md", "", max_chunk_size=10) == []


@pytest.mark.parametrize("size", [0, -1])
def test_split_markdown_rejects_invalid_size(size: int) -> None:
    """Reject invalid sizes even when the document is empty."""
    with pytest.raises(ValueError):
        split_markdown("empty.md", "", max_chunk_size=size)


def test_statement_start_includes_all_decorators() -> None:
    """Start at the first decorator instead of the function definition."""
    code = (
        "DEFAULT_NAME = 'adapter'\n"
        "\n"
        "@require_auth\n"
        "@log_call\n"
        "def load_adapter(path):\n"
        "    return path\n"
    )

    tree = ast.parse(code)
    function = tree.body[1]

    start = statement_start(function, line_starts(code), code)

    assert start == code.index("@require_auth")
    assert code[start:] == (
        "@require_auth\n"
        "@log_call\n"
        "def load_adapter(path):\n"
        "    return path\n"
    )


def test_statement_start_without_decorator() -> None:
    """Start at the definition when the function has no decorator."""
    code = (
        "# Commentaire\n"
        "\n"
        "def load_adapter(path):\n"
        "    return path\n"
    )

    tree = ast.parse(code)
    function = tree.body[0]

    start = statement_start(function, line_starts(code), code)

    assert start == code.index("def load_adapter")


def test_statement_start_includes_adjacent_comments() -> None:
    """Keep consecutive comments above the first decorator."""
    code = (
        "DEFAULT_NAME = 'adapter'\n"
        "\n"
        "# Vérifier les droits.\n"
        "# Journaliser le chargement.\n"
        "@require_auth\n"
        "def load_adapter(path):\n"
        "    return path\n"
    )

    function = ast.parse(code).body[1]

    start = statement_start(function, line_starts(code), code)

    assert start == code.index("# Vérifier")


def test_statement_start_stops_at_blank_line() -> None:
    """Avoid attaching a comment separated by an empty line."""
    code = (
        "# Commentaire général.\n"
        "\n"
        "def load_adapter(path):\n"
        "    return path\n"
    )

    function = ast.parse(code).body[0]

    start = statement_start(function, line_starts(code), code)

    assert start == code.index("def load_adapter")


def test_statement_spans_preserve_module_content() -> None:
    """Partition module instructions without losing source text."""
    code = (
        "# Configuration\n"
        "DEFAULT_NAME = 'adapter'\n"
        "\n"
        "@require_auth\n"
        "def load_adapter(path):\n"
        "    return path\n"
    )

    tree = ast.parse(code)

    spans = statement_spans(code, tree.body)

    boundary = code.index("@require_auth")

    assert spans == [(0, boundary), (boundary, len(code))]
    assert "".join(code[start:end] for start, end in spans) == code


def test_statement_spans_preserve_comments_only() -> None:
    """Keep a non-empty document even when it has no AST statements."""
    code = "# Commentaire général\n\n"
    tree = ast.parse(code)

    assert tree.body == []
    assert statement_spans(code, tree.body) == [(0, len(code))]


def test_statement_spans_handle_empty_document() -> None:
    """Return no blocks for an empty document."""
    assert statement_spans("", ast.parse("").body) == []


def test_split_python_groups_small_module_statements() -> None:
    """Group small statements without losing source text or offsets."""
    code = (
        "import os\n\n"
        "@require_auth\n"
        "def load_adapter(path):\n"
        "    return path\n"
    )

    chunks = split_python("example.py", code, max_chunk_size=200)

    assert len(chunks) == 1
    assert chunks[0].text == code
    assert chunks[0].first_character_index == 0
    assert chunks[0].last_character_index == len(code)


def test_split_python_falls_back_on_invalid_syntax() -> None:
    """Keep malformed Python readable instead of failing the pipeline."""
    code = "def broken(:\n    return True\n"

    chunks = split_python("broken.py", code, max_chunk_size=10)

    assert "".join(chunk.text for chunk in chunks) == code

    for chunk in chunks:
        start = chunk.first_character_index
        end = chunk.last_character_index
        assert chunk.text == code[start:end]
        assert 0 < len(chunk.text) <= 10


def test_pack_spans_groups_small_neighbors() -> None:
    """Combine neighboring blocks that fit within the limit."""
    spans = [(0, 20), (20, 35), (35, 80)]

    assert pack_spans(spans, max_chunk_size=50) == [
        (0, 35),
        (35, 80),
    ]


def test_pack_spans_preserves_oversized_block() -> None:
    """Leave a large block available for later subdivision."""
    spans = [(0, 10), (10, 100), (100, 110)]

    assert pack_spans(spans, max_chunk_size=50) == spans


def test_pack_spans_handles_empty_input() -> None:
    """Return no ranges when no blocks are supplied."""
    assert pack_spans([], max_chunk_size=50) == []


def test_split_python_keeps_function_whole_when_group_cannot_fit() -> None:
    """Separate an import when combining it with a function is too large."""
    prefix = "import os\n\n"
    function = (
        "@require_auth\n"
        "def load_adapter(path):\n"
        "    return path\n"
    )
    code = prefix + function

    chunks = split_python(
        "example.py",
        code,
        max_chunk_size=len(function),
    )

    assert [chunk.text for chunk in chunks] == [prefix, function]
    assert chunks[1].first_character_index == len(prefix)
    assert chunks[1].last_character_index == len(code)


def test_child_statements_return_direct_methods_only() -> None:
    """Inspect one AST level without collecting nested instructions."""
    code = (
        "class AdapterManager:\n"
        "    def load(self):\n"
        "        return True\n"
        "\n"
        "    def unload(self):\n"
        "        return False\n"
    )

    class_node = ast.parse(code).body[0]
    methods = child_statements(class_node)

    assert len(methods) == 2
    assert all(isinstance(node, ast.FunctionDef) for node in methods)

    instructions = child_statements(methods[0])

    assert len(instructions) == 1
    assert isinstance(instructions[0], ast.Return)


def test_statement_spans_stay_within_class_range() -> None:
    """Subdivide a class without including neighboring module content."""
    code = (
        "import os\n\n"
        "class Manager:\n"
        "    def load(self):\n"
        "        return True\n"
        "\n"
        "    def unload(self):\n"
        "        return False\n"
        "\n"
        "OTHER = 1\n"
    )

    class_node = ast.parse(code).body[1]
    first = code.index("class Manager")
    last = code.index("OTHER")
    methods = child_statements(class_node)

    spans = statement_spans(code, methods, span=(first, last))

    assert spans == [
        (first, code.index("    def load")),
        (code.index("    def load"), code.index("    def unload")),
        (code.index("    def unload"), last),
    ]

    assert "".join(code[start:end] for start, end in spans) == (
        code[first:last]
    )


def test_split_python_subdivides_class_at_methods() -> None:
    """Use method boundaries when a whole class exceeds the limit."""
    header = "class Manager:\n"
    first_method = (
        "    def load(self):\n"
        "        return True\n"
        "\n"
    )
    second_method = (
        "    def unload(self):\n"
        "        return False\n"
    )
    code = header + first_method + second_method
    limit = len(header + first_method)

    chunks = split_python("example.py", code, max_chunk_size=limit)

    assert [chunk.text for chunk in chunks] == [
        header + first_method,
        second_method,
    ]

    for chunk in chunks:
        start = chunk.first_character_index
        end = chunk.last_character_index

        assert chunk.text == code[start:end]
        assert chunk.file_path == "example.py"
        assert 0 < len(chunk.text) <= limit


def test_split_python_subdivides_large_assignment() -> None:
    """Fall back to text when a large instruction has no child statements."""
    code = "MODELS = [" + ", ".join(["'adapter'"] * 30) + "]\n"

    chunks = split_python("example.py", code, max_chunk_size=50)

    assert len(chunks) > 1
    assert "".join(chunk.text for chunk in chunks) == code

    for chunk in chunks:
        start = chunk.first_character_index
        end = chunk.last_character_index

        assert chunk.text == code[start:end]
        assert 0 < len(chunk.text) <= 50


@pytest.mark.parametrize(
    ("file_path", "expected_count"),
    [
        ("example.md", 2),
        ("example.txt", 1),
        ("example.py", 1),
    ],
)
def test_chunk_file_selects_strategy(
    file_path: str,
    expected_count: int,
) -> None:
    """Interpret headings according to the selected file type."""
    text = "# Première section\nTexte.\n## Deuxième section\nSuite.\n"

    chunks = chunk_file(file_path, text)

    assert len(chunks) == expected_count
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(chunk.file_path == file_path for chunk in chunks)


@pytest.mark.parametrize("size", [0, -1, 2001])
def test_chunk_file_rejects_invalid_size(size: int) -> None:
    """Enforce the project's chunk-size limits."""
    with pytest.raises(ValueError):
        chunk_file("example.txt", "Hello", max_chunk_size=size)


def test_chunk_file_rejects_unknown_extension() -> None:
    """Report unsupported inputs instead of silently selecting a strategy."""
    with pytest.raises(ValueError):
        chunk_file("example.json", "{}")
