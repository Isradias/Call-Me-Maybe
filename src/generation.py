"""Public-SDK model helpers and prompt construction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast

from .errors import GenerationError
from .models import FunctionDefinitions


class EncodedBatch(Protocol):
    """Describe the public tensor result returned by the SDK encoder."""

    def tolist(self) -> list[list[int]]:
        """Return token IDs as one batch."""


class PublicModel(Protocol):
    """Declare only public methods used from ``llm_sdk``."""

    def encode(self, text: str) -> EncodedBatch:
        """Encode text into token IDs."""

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs into text."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits."""

    def get_path_to_vocab_file(self) -> str:
        """Return the public vocabulary-file path."""


def create_model() -> PublicModel:
    """Create the required Qwen model through the supplied SDK."""
    try:
        from llm_sdk import Small_LLM_Model

        return cast(PublicModel, Small_LLM_Model("Qwen/Qwen3-0.6B"))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise GenerationError(f"Could not initialize Qwen: {error}") from error


def encode_text(model: PublicModel, text: str) -> list[int]:
    """Encode one non-empty prompt through the public SDK method."""
    batches = model.encode(text).tolist()
    if len(batches) != 1 or not batches[0]:
        raise GenerationError("The SDK encoder returned no usable token IDs.")
    return list(batches[0])


def load_vocabulary(model: PublicModel) -> dict[str, int]:
    """Read the public token-to-ID vocabulary mapping."""
    try:
        with Path(model.get_path_to_vocab_file()).open("r", encoding="utf-8") as source:
            vocabulary = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationError(f"Could not load model vocabulary: {error}") from error
    if not isinstance(vocabulary, dict) or not all(
        isinstance(token, str) and isinstance(token_id, int)
        for token, token_id in vocabulary.items()
    ):
        raise GenerationError("Model vocabulary has an invalid format.")
    return vocabulary


def build_prompt(request: str, functions: FunctionDefinitions) -> str:
    """Build an instruction for one schema-valid function-call JSON object."""
    definitions = json.dumps(
        [function.model_dump() for function in functions.root],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return (
        "Choose exactly one function and extract its arguments. "
        "Return only JSON with this exact shape: "
        '{"name":"function_name","parameters":{...}}. '
        "Use a function with additional behavior only when the request explicitly "
        "asks for that behavior.\n"
        f"Functions:{definitions}\n"
        f"Request:{request}\n"
        "JSON:"
    )
