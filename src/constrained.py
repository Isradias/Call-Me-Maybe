"""Schema-aware, token-by-token constrained JSON decoding."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict

from .errors import GenerationError
from .generation import PublicModel, encode_text
from .models import FunctionDefinition, FunctionDefinitions


INCOMPLETE = -1
INVALID = -2
WHITESPACE = " \t\r\n"


class PrefixStatus(BaseModel):
    """Describe whether a generated JSON prefix can still become valid."""

    model_config = ConfigDict(frozen=True)

    possible: bool
    complete: bool


def _space(text: str, index: int) -> int:
    while index < len(text) and text[index] in WHITESPACE:
        index += 1
    return index


def _literal(text: str, index: int, expected: str) -> int:
    remaining = text[index:]
    if text.startswith(expected, index):
        return index + len(expected)
    return INCOMPLETE if expected.startswith(remaining) else INVALID


def _string(text: str, index: int) -> int:
    if index == len(text):
        return INCOMPLETE
    if text[index] != '"':
        return INVALID
    index += 1
    while index < len(text):
        character = text[index]
        if character == '"':
            return index + 1
        if ord(character) < 32:
            return INVALID
        if character != "\\":
            index += 1
            continue
        index += 1
        if index == len(text):
            return INCOMPLETE
        if text[index] in '"\\/bfnrt':
            index += 1
            continue
        if text[index] != "u":
            return INVALID
        index += 1
        for _ in range(4):
            if index == len(text):
                return INCOMPLETE
            if text[index] not in "0123456789abcdefABCDEF":
                return INVALID
            index += 1
    return INCOMPLETE


def _number(text: str, index: int) -> int:
    if index < len(text) and text[index] == "-":
        index += 1
        if index == len(text):
            return INCOMPLETE
    if index == len(text) or not text[index].isdigit():
        return INVALID
    if text[index] == "0":
        index += 1
    else:
        while index < len(text) and text[index].isdigit():
            index += 1
    if index < len(text) and text[index] == ".":
        index += 1
        if index == len(text):
            return INCOMPLETE
        if not text[index].isdigit():
            return INVALID
        while index < len(text) and text[index].isdigit():
            index += 1
    if index < len(text) and text[index] in "eE":
        index += 1
        if index < len(text) and text[index] in "+-":
            index += 1
        if index == len(text):
            return INCOMPLETE
        if not text[index].isdigit():
            return INVALID
        while index < len(text) and text[index].isdigit():
            index += 1
    return index


def _value(text: str, index: int) -> int:
    index = _space(text, index)
    if index == len(text):
        return INCOMPLETE
    character = text[index]
    if character == '"':
        return _string(text, index)
    if character in "-0123456789":
        return _number(text, index)
    if character == "t":
        return _literal(text, index, "true")
    if character == "f":
        return _literal(text, index, "false")
    if character == "n":
        return _literal(text, index, "null")
    if character == "{":
        return _object(text, index)
    if character == "[":
        return _array(text, index)
    return INVALID


def _object(text: str, index: int) -> int:
    index = _space(text, index + 1)
    if index == len(text):
        return INCOMPLETE
    if text[index] == "}":
        return index + 1
    while True:
        index = _string(text, index)
        if index < 0:
            return index
        index = _space(text, index)
        index = _literal(text, index, ":")
        if index < 0:
            return index
        index = _value(text, index)
        if index < 0:
            return index
        index = _space(text, index)
        if index == len(text):
            return INCOMPLETE
        if text[index] == "}":
            return index + 1
        if text[index] != ",":
            return INVALID
        index = _space(text, index + 1)
        if index == len(text) or text[index] == "}":
            return INCOMPLETE if index == len(text) else INVALID


def _array(text: str, index: int) -> int:
    index = _space(text, index + 1)
    if index == len(text):
        return INCOMPLETE
    if text[index] == "]":
        return index + 1
    while True:
        index = _value(text, index)
        if index < 0:
            return index
        index = _space(text, index)
        if index == len(text):
            return INCOMPLETE
        if text[index] == "]":
            return index + 1
        if text[index] != ",":
            return INVALID
        index = _space(text, index + 1)
        if index == len(text) or text[index] == "]":
            return INCOMPLETE if index == len(text) else INVALID


def _typed_value(text: str, index: int, type_name: str) -> int:
    index = _space(text, index)
    if index == len(text):
        return INCOMPLETE
    if type_name == "string":
        return _string(text, index)
    if type_name in {"number", "integer"}:
        result = _number(text, index)
        if type_name == "integer" and result >= 0 and result < len(text):
            if text[result] in ".eE":
                return INVALID
        return result
    if type_name == "boolean":
        true_end = _literal(text, index, "true")
        false_end = _literal(text, index, "false")
        return max(true_end, false_end)
    if type_name == "null":
        return _literal(text, index, "null")
    if type_name == "object":
        return _object(text, index) if text[index] == "{" else INVALID
    if type_name == "array":
        return _array(text, index) if text[index] == "[" else INVALID
    return INVALID


def _choice(text: str, index: int, choices: dict[str, str]) -> tuple[int, str | None]:
    for literal, value in choices.items():
        if text.startswith(literal, index):
            return index + len(literal), value
    remaining = text[index:]
    if any(literal.startswith(remaining) for literal in choices):
        return INCOMPLETE, None
    return INVALID, None


def _parameters(text: str, index: int, function: FunctionDefinition) -> int:
    if index == len(text):
        return INCOMPLETE
    if text[index] != "{":
        return INVALID
    index = _space(text, index + 1)
    seen: set[str] = set()
    if index == len(text):
        return INCOMPLETE
    if text[index] == "}":
        return index + 1 if not function.parameters else INVALID
    while True:
        choices = {
            json.dumps(name, ensure_ascii=False): name
            for name in function.parameters
            if name not in seen
        }
        index, parameter = _choice(text, index, choices)
        if index < 0 or parameter is None:
            return index
        seen.add(parameter)
        index = _space(text, index)
        index = _literal(text, index, ":")
        if index < 0:
            return index
        index = _typed_value(text, index, function.parameters[parameter].type)
        if index < 0:
            return index
        index = _space(text, index)
        if index == len(text):
            return INCOMPLETE
        if text[index] == "}":
            return index + 1 if len(seen) == len(function.parameters) else INVALID
        if text[index] != "," or len(seen) == len(function.parameters):
            return INVALID
        index = _space(text, index + 1)
        if index == len(text):
            return INCOMPLETE


def validate_prefix(prefix: str, functions: FunctionDefinitions) -> PrefixStatus:
    """Check whether a prefix can become one complete schema-valid call object."""
    index = _space(prefix, 0)
    for literal in ("{", '"name"', ":"):
        index = _literal(prefix, _space(prefix, index), literal)
        if index < 0:
            return PrefixStatus(possible=index == INCOMPLETE, complete=False)
    names = {json.dumps(function.name): function.name for function in functions.root}
    index, name = _choice(prefix, _space(prefix, index), names)
    if index < 0:
        return PrefixStatus(possible=index == INCOMPLETE, complete=False)
    if name is None:
        return PrefixStatus(possible=False, complete=False)
    try:
        function = functions.by_name(name)
    except KeyError:
        return PrefixStatus(possible=False, complete=False)
    for literal in (",", '"parameters"', ":"):
        index = _literal(prefix, _space(prefix, index), literal)
        if index < 0:
            return PrefixStatus(possible=index == INCOMPLETE, complete=False)
    index = _parameters(prefix, _space(prefix, index), function)
    if index < 0:
        return PrefixStatus(possible=index == INCOMPLETE, complete=False)
    index = _literal(prefix, _space(prefix, index), "}")
    if index < 0:
        return PrefixStatus(possible=index == INCOMPLETE, complete=False)
    index = _space(prefix, index)
    return PrefixStatus(
        possible=index == len(prefix),
        complete=index == len(prefix),
    )


def generate_call_json(
    model: PublicModel,
    prompt: str,
    functions: FunctionDefinitions,
    vocabulary: dict[str, int],
    max_new_tokens: int = 128,
) -> str:
    """Generate a complete JSON call while rejecting invalid schema prefixes."""
    ids = encode_text(model, prompt)
    prefix = ""
    cache: dict[int, str] = {}
    vocabulary_ids = set(vocabulary.values())
    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(ids)
        valid_id: int | None = None
        for token_id in sorted(vocabulary_ids, key=lambda item: logits[item], reverse=True):
            token = cache.get(token_id)
            if token is None:
                token = model.decode([token_id])
                cache[token_id] = token
            if token and validate_prefix(prefix + token, functions).possible:
                valid_id = token_id
                break
        if valid_id is None:
            raise GenerationError("No token can preserve the function-call schema.")
        token = cache[valid_id]
        prefix += token
        ids.append(valid_id)
        if validate_prefix(prefix, functions).complete:
            return prefix
    raise GenerationError("Generation reached the token limit before valid JSON completed.")
