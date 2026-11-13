"""Tests for lexical tokenization and identifier expansion."""

import pytest

from src.tokenizer import tokenize


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("load_adapter", ["load_adapter", "load", "adapter"]),
        ("max_model_len", ["max_model_len", "max", "model", "len"]),
        ("adapter", ["adapter"]),
        ("_load__adapter", ["_load__adapter", "load", "adapter"]),
        ("adapter adapter", ["adapter", "adapter"]),
        ("LoRA", ["lora", "lo", "ra"]),
        ("", []),
        (" / ? ! ", []),
    ],
)
def test_tokenize_terms(text: str, expected: list[str]) -> None:
    """Preserve whole terms and expand snake_case consistently."""
    assert tokenize(text) == expected


def test_tokenize_endpoint() -> None:
    """Separate path delimiters while expanding the identifier."""
    assert tokenize("POST /v1/load_lora_adapter") == [
        "post",
        "v1",
        "load_lora_adapter",
        "load",
        "lora",
        "adapter",
    ]
