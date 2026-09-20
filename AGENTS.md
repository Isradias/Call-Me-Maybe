# Repository Guidelines

## Project Structure & Module Organization

- `main.py` defines `Function_Calling`, which selects a function from the catalog and extracts its parameters.
- `llm_sdk/__init__.py` contains `Small_LLM_Model`, the local Hugging Face model and tokenizer wrapper.
- `data/input/functions_definition.json` is the source of truth for callable function names, descriptions, parameter schemas, and return types.
- `data/input/function_calling_tests.json` provides representative natural-language prompts for manual evaluation.
- `teste.py` is an exploratory script for inspecting the tokenizer vocabulary. Keep ad hoc experiments separate from runtime code.

## Setup, Development, and Verification

Use Python 3.12 or newer. This repository uses `uv` and commits `uv.lock` for reproducible dependency resolution.

```bash
uv sync                 # create/update the environment and install dependencies
uv run python main.py   # run the application entry point when one is present
uv run python -m py_compile main.py llm_sdk/__init__.py  # syntax check
uv run python teste.py  # inspect the configured model's vocabulary
```

The first model run may download Hugging Face model artifacts. Use the prompts in `data/input/function_calling_tests.json` as smoke-test cases after changing selection or extraction behavior.

## Coding Style & Naming Conventions

Use four spaces for indentation, UTF-8 files, type annotations for public methods, and concise docstrings for reusable model-wrapper APIs. Keep imports grouped at the top and remove unused imports. Existing code uses `snake_case` for methods, variables, JSON fields, and function-catalog names (for example, `fn_add_numbers`); preserve that convention. Class names use `PascalCase`.

Keep function definitions valid JSON. Add a clear `description`, declare every parameter's `type`, and use stable `fn_<verb>_<object>` names so constrained decoding can match exact prefixes.

## Testing Guidelines

There is no automated test framework or coverage target yet. For each behavior change, run the syntax check above and manually exercise relevant prompts from the JSON fixture. Verify both the chosen function name and parameter order/value, especially for similar functions such as `fn_add_numbers` and `fn_add_numbers_squares`.

## Commit & Pull Request Guidelines

Recent history uses short, imperative feature subjects, commonly `Feat: <change>` (for example, `Feat: get_output_parameters`). Keep commits focused and describe observable behavior. Pull requests should explain the prompt or schema change, identify affected catalog entries, include commands run, and show before/after output for inference changes. Link related issues when available.
