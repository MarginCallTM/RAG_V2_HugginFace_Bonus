"""Generate a first grounded answer with Qwen on CPU"""

from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.indexer import load_index
from src.prompting import build_budgeted_prompt
from src.retriever import Retriever


def main() -> None:
    """Retrieve evidence, prepare the prompt and generate an answer."""
    model_name = "Qwen/Qwen3-0.6B"
    context_budget = 4096
    max_new_tokens = 256

    question = (
        "What HTTP endpoint is used to dynamically load "
        "a LoRA adapter in vLLM?"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    index = load_index(Path("data/processed"))
    retriever = Retriever(index)
    hits = retriever.search(question, k=3)
    chunks = [hit.chunk for hit in hits]

    prompt = build_budgeted_prompt(
        question,
        chunks,
        tokenizer,
        context_budget=context_budget,
        max_new_tokens=max_new_tokens,
    )

    messages = [
        message.model_dump()
        for message in prompt.messages
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenizer=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    input_tokens = inputs["input_ids"].shape[-1]

    if input_tokens + max_new_tokens > context_budget:
        raise ValueError("Prepared input exceeds the application budget")

    print("\nLoading Qwen on CPU...", flush=True)
    loading_started = perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )
    model.cpu()
    model.eval()

    loading_seconds = perf_counter() - loading_started

    print("Generating answer...", flush=True)
    generation_started = perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_seconds = perf_counter() - generation_started

    generated_ids = outputs[0, input_tokens:]
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    print("\nQUESTION")
    print(question)

    print("\nANSWER")
    print(answer)

    print("\nINCLUDED SOURCES")
    for number, source in enumerate(prompt.sources, start=1):
        print(f"Source {number}:", source.model_dump())

    print("\nMEASUREMENTS")
    print("Input tokens:", input_tokens)
    print("Generated tokens:", len(generated_ids))
    print("Loading seconds:", f"{loading_seconds:.2f}")
    print("Generation seconds:", f"{generation_seconds:.2f}")
    print(
        "Output token limit reached:",
        len(generated_ids) == max_new_tokens

    )


if __name__ == "__main__":
    main()
