"""Discover and read files from the source corpus."""

from pathlib import Path

DEFAULT_EXTENSIONS = (".py", ".txt", ".md")


def list_corpus_files(
    raw_dir: Path,
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
) -> list[Path]:
    """Return matching corpus files in deterministic path order.

    Args:
        raw_dir: Directory to explore recursively.
        extensions: File extensions to retain, including the leading dot.

    Returns:
        Matching file paths sorted by path.

    Raises:
        NotADirectoryError: If raw_dir is not an existing directory.
    """
    if not raw_dir.is_dir():
        raise NotADirectoryError(
            f"Corpus directory not found: {raw_dir}"
        )
    files = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix in extensions
    ]

    return sorted(files)


def read_text(path: Path) -> str:
    """Read a corpus file as UTF-8 text with normalized line endings.

    Args:
        Decoded text using python's default newline normalization.

    Raises:
        OSERROR: If the file cannot be opened or read.
        UnicodeDecodeError: If its content is not valid UTF-8
    """
    with path.open(encoding="utf-8") as handle:
        return handle.read()


def to_corpus_path(path: Path, project_root: Path) -> str:
    """Express a source path relative to the project root.

    Args:
        path: Path of a source file.
        project_root: Root directory of the project.

    Returns:
        Relative source path using forward slashes.

    Raises:
        ValueError: If the resolved path is outside the project root.
    """
    resolved_path = path.resolve()
    resolved_root = project_root.resolve()

    return resolved_path.relative_to(resolved_root).as_posix()
