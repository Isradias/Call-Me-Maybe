"""Safe JSON loading and Pydantic validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import RootModel, ValidationError

from .errors import InputError
from .models import FunctionDefinitions, PromptDefinitions


ModelT = TypeVar("ModelT", bound=RootModel[Any])


def load_json(path: Path, label: str) -> Any:
    """Load a UTF-8 JSON document and provide a useful error on failure."""
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError as error:
        raise InputError(f"{label} file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise InputError(f"Invalid JSON in {label}: {error.msg}") from error
    except OSError as error:
        raise InputError(f"Could not read {label}: {path}") from error


def load_functions(path: Path) -> FunctionDefinitions:
    """Load and validate the function-definition document."""
    try:
        return FunctionDefinitions.model_validate(load_json(path, "function definitions"))
    except ValidationError as error:
        raise InputError(f"Invalid function-definition schema: {error}") from error


def load_prompts(path: Path) -> PromptDefinitions:
    """Load and validate the prompt-input document."""
    try:
        return PromptDefinitions.model_validate(load_json(path, "prompt input"))
    except ValidationError as error:
        raise InputError(f"Invalid prompt-input schema: {error}") from error
