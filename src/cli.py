"""Command-line interface for the function-calling application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import GenerationError, InputError
from .loader import load_functions, load_prompts
from .workflow import run_workflow


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse paths required by the project command-line interface."""
    parser = argparse.ArgumentParser(
        description="Generate schema-valid function calls with Qwen/Qwen3-0.6B."
    )
    parser.add_argument("--functions_definition", type=Path,
                        default=Path("data/input/functions_definition.json"))
    parser.add_argument("--input", type=Path,
                        default=Path("data/input/function_calling_tests.json"))
    parser.add_argument("--output", "-output", type=Path,
                        default=Path("data/output/function_calling_results.json"))
    return parser.parse_args(arguments)


def run(arguments: list[str] | None = None) -> int:
    """Run the complete workflow and return an appropriate shell status."""
    options = parse_arguments(arguments)
    try:
        functions = load_functions(options.functions_definition)
        prompts = load_prompts(options.input)
        results = run_workflow(functions, prompts, options.output)
    except (InputError, GenerationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Generated {len(results)} function call(s): {options.output}")
    return 0
