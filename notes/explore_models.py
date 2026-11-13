"""Explore source validation and JSON serialization"""

from pydantic import ValidationError

from src.models import MinimalSource


def main() -> None:
    """Create a valid source and attempt to create an invalid one."""
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
        first_character_index=4695,
        last_character_index=6098,
    )

    print("OBJECT :", source)
    print("DEBUT :", source.first_character_index)
    print("JSON :", source.model_dump_json(indent=2))

    try:
        MinimalSource(
            file_path="example.py",
            first_character_index=10,
            last_character_index=5,
        )
    except ValidationError as error:
        print("\nSOURCE REFUSEE :")
        print(error)


if __name__ == "__main__":
    main()
