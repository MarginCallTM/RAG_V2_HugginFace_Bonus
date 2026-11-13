"""Observe how text reading handles line endings."""

from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Compare default reading with preserved line endings."""
    with TemporaryDirectory() as directory:
        path = Path(directory) / "exemple.txt"
        path.write_bytes(b"ABC\r\nDEF")

        with path.open(encoding="utf-8") as handle:
            normalized = handle.read()

        with path.open(encoding="utf=8", newline="") as handle:
            preserved = handle.read()

        print("LECTURE PAR DEFAUT :", repr(normalized))
        print("LONGUEUR :", len(normalized))
        print("POSITION DE D :", normalized.index("D"))

        print("\nFINS DE LIGNE CONSERVEES :", repr(preserved))
        print("LONGUEUR :", len(preserved))
        print("POSITION DE D :", preserved.index("D"))


if __name__ == "__main__":
    main()
