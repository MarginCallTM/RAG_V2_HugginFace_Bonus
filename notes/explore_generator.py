"""Excercise the reusable generator with retrieved evidence."""

from pathlib import Path

from src.generator import Generator
from src.indexer import load_index
from src.prompting import build_budgeted_prompt
from src.retriever import Retriever


def main() -> None:
    """Generate a grounded answer with the reusable components."""
    question = (
        "What HTTP endpoint is used to dynamically load "
        "a LoRA adapter in vLLM?"
    )

    retriever = Retriever(load_index(Path("data/processed")))
    chunks = [
        hit.chunk
        for hit in retriever.search(question, k=3)
    ]

    print("Loading Qwen...", flush=True)
    generator = Generator()

    print("Device:", generator.device)
    print("Model dtype:", generator.model.dtype)
    print("Selected device:", generator.device)
    print("Model device:", generator.model.device)
    print("Model dtype:", generator.model.dtype)

    prompt = build_budgeted_prompt(
        question,
        chunks,
        generator.tokenizer,
        context_budget=generator.context_budget,
        max_new_tokens=generator.max_new_tokens,
    )

    print("Generating...", flush=True)
    answer = generator.generate(prompt)

    print("\nANSWER")
    print(answer)


if __name__ == "__main__":
    main()
