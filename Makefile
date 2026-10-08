.PHONY: install run debug clean lint lint-strict test

ARGS ?=

install:
	uv sync

run:
	uv run python -m src $(ARGS)

debug:
	uv run python -m pdb -m src $(ARGS)

clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache
	rm -rf data/output

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run python -m unittest discover -s tests -v
