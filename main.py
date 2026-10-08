"""Backward-compatible entry point for the ``src`` application package."""

from src.cli import run


def main() -> None:
    """Run the command-line application."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
