"""Inspect retrieved evidence and a draft prompt wihtout loading a model."""

from pathlib import Path

from src.indexer import load_index
from src.retriever import Retriever
from src.prompting import (
    build_prompt,
    build_budgeted_prompt,
    count_prompt_tokens
)
from transformers import AutoTokenizer


def main() -> None:
    """Display three ranked passages with their exact provenance."""
    question = (
        "What HTTP endpoint is used to dynamically load "
        "a LoRA adapter in vLLM?"
    )
    index = load_index(Path("data/processed"))
    retriever = Retriever(index)
    hits = retriever.search(question, k=3)

    # Extract the chunks from the ranked BM25 results.
    chunks = [hit.chunk for hit in hits]

    # Build the messages and keep their associated sources.
    prompt = build_prompt(question, chunks)

    for message in prompt.messages:
        print(f"\n{message.role.upper()} MESSAGE\n")
        print(message.content)

    print("\nSources included:", len(prompt.sources))

    for number, source in enumerate(prompt.sources, start=1):
        print(f"Source {number}:", source.model_dump())

    # Load tokenizer files, without loading model weights.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Convert our Pydantic messages into dictionaries.
    messages = [
        message.model_dump()
        for message in prompt.messages
    ]

    # Format and tokenize the complete conversation.
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    context_budget = 4096
    max_new_tokens = 512
    input_budget = context_budget - max_new_tokens
    input_tokens = len(token_ids)

    print("\nTOKEN BUDGET")
    print("Aplication context budget:", context_budget)
    print("Reserved output tokens:", max_new_tokens)
    print("Available input tokens", input_budget)
    print("Actual input tokens:", input_tokens)
    print("Fits application budget: ", input_tokens <= input_budget)

    print("\nFIRST 20 TOKEN IDS")
    print(token_ids[:20])

    print("\nFORMATTED PROMPT PREVIEW")
    print(tokenizer.decode(token_ids[:100]))

    print("\nBUDGET SELECTION EXPERIMENT")

    for budget in (4096, 1024, 512):
        bounded = build_budgeted_prompt(
            question,
            chunks,
            tokenizer,
            context_budget=budget,
            max_new_tokens=256,
        )

        measured = count_prompt_tokens(bounded, tokenizer)

        print(
            f"Total budget={budget}, "
            f"input limit={budget - 256}"
            f"input tokens={measured}, "
            f"sources={len(bounded.sources)}"
        )


if __name__ == "__main__":
    main()
