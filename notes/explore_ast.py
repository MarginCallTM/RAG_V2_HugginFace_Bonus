"""Inspect Python synthax nodes and their line positions"""

import ast

from src.chunking import line_starts, statement_start, statement_spans


def main() -> None:
    """Display top-level statements and function decorators"""
    code = (
        "# Gestion des adaptateurs\n"
        "DEFAULT_NAME = 'adapter'\n"
        "\n"
        "@require_auth\n"
        "def load_adapter(path):\n"
        "   validate_path(path)\n"
        "   return True\n"
    )

    print("CODE SOURCE :")
    for number, line in enumerate(code.splitlines(), start=1):
        print(f"{number}: {line}")

    tree = ast.parse(code)
    starts = line_starts(code)

    print("\nPOSITIONS DES DEBUTS DE LIGNES :")
    for number, position in enumerate(starts, start=1):
        print(f"ligne {number} -> caractere {position}")

    print("\nINSTRUCTIONS AU NIVEAU DU MODULE :")

    for node in tree.body:
        print(
            type(node).__name__,
            "debut :", node.lineno,
            "fin :", node.end_lineno,
        )

        if isinstance(node, ast.FunctionDef):
            print("NOM :", node.name)

            first_character = statement_start(node, starts, code)

            print("POSITION DE DEBUT :", first_character)
            print("TEXTE DEPUIS CETTE POSITION :")
            print(code[first_character:])
    print("\nBLOCS DU MODULE: ")

    for start, end in statement_spans(code, tree.body):
        print(f"\n[{start}:{end}]")
        print(repr(code[start:end]))


if __name__ == "__main__":
    main()
