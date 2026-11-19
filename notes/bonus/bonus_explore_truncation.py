"""Observe information lost when passages are truncated."""

from transformers import AutoTokenizer


def main() -> None:
    """Compare distinct passages before and after token truncation"""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prefix = "cache " * 280

    passages = [
        prefix + "The HTTP server listens on port 8000.",
        prefix + "The HTTP server listens on port 9000.",
    ]

    retained_ids: list[list[int]] = []

    print("Truncation side:", tokenizer.truncation_side)

    for index, passage in enumerate(passages):
        full = tokenizer(
            passage,
            truncation=False,
        )

        truncated = tokenizer(
            passage,
            truncation=True,
            max_length=256,
        )

        full_ids = full["input_ids"]
        print("First tokens:", tokenizer.convert_ids_to_tokens(full_ids[:15]))
        kept_ids = truncated["input_ids"]
        retained_ids.append(kept_ids)

        print(f"\nPassage {index}")
        print("Characters:", len(passage))
        print("Within 2000 characters:", len(passage) <= 2000)
        print("Tokens before truncation:", len(full_ids))
        print("Tokens after truncation:", len(kept_ids))
        print("Original ending:", passage[-80:])
        print(
            "Retained ending:",
            tokenizer.decode(
                kept_ids[-20:],
                skip_special_tokens=True,
            ),
        )

    print("\nOriginal texts identical:", passages[0] == passages[1])
    print("Retained token IDs identical:", retained_ids[0] == retained_ids[1])


if __name__ == "__main__":
    main()
