"""Observe lexical tokenization before retrieval scoring"""

from src.tokenizer import tokenize, split_identifier


def main() -> None:
    """Compare natural-language terms and Python identifiers"""
    examples = [
        "load_adapter",
        "max_model_len",
        "adapter",
        "_load__adapter",
        "adapter adapter",
        "POST /v1/load_lora_adapter",
        "AdapterManager",
        "LLMEngine",
        "LoRA",
    ]

    for text in examples:
        print("\nTEXTE :", text)
        print("TERMES :", tokenize(text))

    print("\nEXPLORATION DES IDENTIFIANTS :")

    for identifier in (
        "AdapterManager",
        "LLMEngine",
        "get_model_config",
        "Qwen3",
        "LoRA",
        "vLLM",
    ):
        print(identifier, "→", split_identifier(identifier))


if __name__ == "__main__":
    main()
