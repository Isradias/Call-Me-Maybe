"""End-to-end calling workflow and atomic result writing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .constrained import generate_call_json
from .errors import GenerationError, InputError
from .generation import PublicModel, build_prompt, create_model, load_vocabulary
from .models import FunctionCallResult, FunctionDefinitions, PromptDefinitions


def matches_type(value: Any, type_name: str) -> bool:
    """Check a decoded value against one declared top-level JSON type."""
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "null":
        return value is None
    if type_name == "object":
        return isinstance(value, dict)
    return isinstance(value, list) if type_name == "array" else False


def validate_call(
    generated: str, prompt: str, functions: FunctionDefinitions
) -> FunctionCallResult:
    """Parse and defensively validate a generated call after decoding."""
    try:
        payload = json.loads(generated)
    except json.JSONDecodeError as error:
        raise GenerationError(f"Decoder returned invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict) or set(payload) != {"name", "parameters"}:
        raise GenerationError("Generated call must contain only name and parameters.")
    name = payload["name"]
    parameters = payload["parameters"]
    if not isinstance(name, str) or not isinstance(parameters, dict):
        raise GenerationError("Generated call has invalid name or parameters.")
    try:
        function = functions.by_name(name)
    except KeyError as error:
        raise GenerationError("Generated call selected an unknown function.") from error
    if set(parameters) != set(function.parameters):
        raise GenerationError("Generated call has missing or extra parameters.")
    normalized_parameters = dict(parameters)
    for parameter, schema in function.parameters.items():
        if not matches_type(parameters[parameter], schema.type):
            raise GenerationError(f"Parameter '{parameter}' has an invalid type.")
        if schema.type == "number":
            normalized_parameters[parameter] = float(parameters[parameter])
    return FunctionCallResult(
        prompt=prompt,
        name=name,
        parameters=normalized_parameters,
    )


def process_prompts(
    model: PublicModel,
    functions: FunctionDefinitions,
    prompts: PromptDefinitions,
) -> list[FunctionCallResult]:
    """Generate and validate one function call for every prompt."""
    vocabulary = load_vocabulary(model)
    results: list[FunctionCallResult] = []
    for index, prompt_definition in enumerate(prompts.root, start=1):
        try:
            generated = generate_call_json(
                model,
                build_prompt(prompt_definition.prompt, functions),
                functions,
                vocabulary,
            )
            results.append(validate_call(generated, prompt_definition.prompt, functions))
        except GenerationError as error:
            raise GenerationError(f"Prompt {index} failed: {error}") from error
    return results


def write_results(results: list[FunctionCallResult], output_path: Path) -> None:
    """Atomically replace the output document only after all prompts succeed."""
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump([item.model_dump() for item in results], temporary,
                      ensure_ascii=False, indent=4)
            temporary.write("\n")
        os.replace(temporary_path, output_path)
    except OSError as error:
        raise InputError(f"Could not write output file: {output_path}") from error
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def run_workflow(
    functions: FunctionDefinitions,
    prompts: PromptDefinitions,
    output_path: Path,
) -> list[FunctionCallResult]:
    """Initialize the model once, process all prompts, and write results."""
    results = process_prompts(create_model(), functions, prompts)
    write_results(results, output_path)
    return results
