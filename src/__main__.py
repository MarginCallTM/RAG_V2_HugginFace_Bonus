"""Entry point for the RAG application."""
import sys

import fire

from src.cli import RagCli, CliError


def main() -> None:
    """Display commands and present expected errors without tracebacks."""
    try:
        fire.Fire(RagCli)
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
