"""Convert source passages and queries into lexical search terms."""

import re

WORD = re.compile(r"[A-Za-z0-9_]+")

CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z0-9]+|[A-Z]+")


def split_identifier(word: str) -> list[str]:
    """Split underscores and capitalized components before lowercasing."""
    pieces: list[str] = []

    for part in word.split("_"):
        pieces.extend(
            match.group().lower()
            for match in CAMEL.finditer(part)
        )

    return pieces


def tokenize(text: str) -> list[str]:
    """Keep whole terms and add non-empty snake_case components."""
    terms: list[str] = []

    for match in WORD.finditer(text):
        word = match.group()
        lowered = word.lower()
        terms.append(lowered)

        pieces = split_identifier(word)

        if pieces and pieces != [lowered]:
            terms.extend(pieces)
    return terms
