"""Track corpus file contents for incremental indexing."""

from pydantic import BaseModel, Field, model_validator
from hashlib import sha256
from pathlib import Path

from src.ingest import list_corpus_files, read_text, to_corpus_path


class FileFingerprint(BaseModel):
    """Associate a corpus file path with its content fingerprint"""

    file_path: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class CorpusManifest(BaseModel):
    """Store corpus file fingerprints with unique source paths."""

    files: list[FileFingerprint]

    @model_validator(mode="after")
    def check_unique_paths(self) -> "CorpusManifest":
        """Reject duplicate file paths."""
        seen: set[str] = set()

        for file in self.files:
            if file.file_path in seen:
                raise ValueError(
                    f"Duplicate file path: {file.file_path}"
                )
            seen.add(file.file_path)

        return self


class ManifestDiff(BaseModel):
    """Describe file changes between two corpus manifests."""

    unchanged: list[str]
    added: list[str]
    modified: list[str]
    deleted: list[str]


def fingerprint(text: str) -> str:
    """Compute a SHA-256 fingerprint from UTF-8 text."""
    content = text.encode("utf-8")
    return sha256(content).hexdigest()


def build_manifest(
    raw_dir: Path,
    project_root: Path,
) -> CorpusManifest:
    """Build a file inventory using the corpus ingestion convention."""
    files: list[FileFingerprint] = []

    for path in list_corpus_files(raw_dir):
        source_path = to_corpus_path(path, project_root)
        text = read_text(path)

        file_state = FileFingerprint(
            file_path=source_path,
            content_hash=fingerprint(text),
        )
        files.append(file_state)

    return CorpusManifest(files=files)


def compare_manifest(
    previous: CorpusManifest,
    current: CorpusManifest,
) -> ManifestDiff:
    """Classify files by comparing their paths and content hashes."""
    previous_hashes = {
        file.file_path: file.content_hash
        for file in previous.files
    }
    current_hashes = {
        file.file_path: file.content_hash
        for file in current.files
    }

    unchanged: list[str] = []
    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []

    for path, content_hash in current_hashes.items():
        if path not in previous_hashes:
            added.append(path)
        elif content_hash != previous_hashes[path]:
            modified.append(path)
        else:
            unchanged.append(path)

    for path in previous_hashes:
        if path not in current_hashes:
            deleted.append(path)

    return ManifestDiff(
        unchanged=sorted(unchanged),
        added=sorted(added),
        modified=sorted(modified),
        deleted=sorted(deleted),
    )
