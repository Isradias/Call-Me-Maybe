"""Qwen ChatML version of the constrained function-calling prototype.

This file is intentionally separate from ``main.py`` so the two prompt formats
can be compared. It uses only the public ``llm_sdk`` API.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from llm_sdk import Small_LLM_Model


class FunctionCalling(BaseModel):
    """Choose functions and extract arguments with constrained decoding."""

    model_name: str = Field(default="Qwen/Qwen3-0.6B")
    path_fn_definition: Path
    max_tokens_per_value: int = Field(default=32, ge=1)
    _llm: Small_LLM_Model | None = PrivateAttr(default=None)
    _token_cache: dict[int, str] = PrivateAttr(default_factory=dict)
    _functions_definition: list[dict[str, Any]] = PrivateAttr(
        default_factory=list
    )
    _function_names: set[str] = PrivateAttr(default_factory=set)
    _function_prefixes: set[str] = PrivateAttr(default_factory=set)

    def model_post_init(self, __context: Any) -> None:
        """Load the model and the function schema after Pydantic validation."""
        self._llm = Small_LLM_Model(self.model_name)
        self.load_functions()

    @property
    def llm(self) -> Small_LLM_Model:
        """Return the initialized model wrapper."""
        if self._llm is None:
            raise RuntimeError("The language model was not initialized.")
        return self._llm

    def load_functions(self) -> None:
        """Load function definitions and build valid-name prefix sets."""
        try:
            with self.path_fn_definition.open("r", encoding="utf-8") as file:
                definitions = json.load(file)
        except FileNotFoundError as error:
            raise ValueError(
                "Function definition file not found: "
                f"{self.path_fn_definition}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in function definition file: {error.msg}"
            ) from error

        if not isinstance(definitions, list):
            raise ValueError("Function definitions must be a JSON array.")

        self._functions_definition = definitions
        self._function_names = {
            definition["name"]
            for definition in definitions
            if isinstance(definition, dict)
            and isinstance(definition.get("name"), str)
        }
        if not self._function_names:
            raise ValueError(
                "Function definitions do not contain any valid names."
            )

        self._function_prefixes = {
            name[:index]
            for name in self._function_names
            for index in range(1, len(name) + 1)
        }

    def get_function_definition(self, name: str) -> dict[str, Any]:
        """Return the schema for a known function name."""
        for definition in self._functions_definition:
            if definition.get("name") == name:
                return definition
        raise ValueError(f"Unknown function name: {name}")

    def selection_system_message(self) -> str:
        """Build the Qwen system instruction for function selection."""
        definitions = json.dumps(
            self._functions_definition, ensure_ascii=False
        )
        return (
            "You are a function-calling system. Select exactly one function "
            "whose description best matches the user request. Return only its "
            "exact name inside double quotes, with no explanation. Compare all "
            "descriptions before choosing. Prefer the least specific function "
            "that fully satisfies the request. Select a function with extra "
            "transformations or conditions only when the user explicitly "
            "requests that behavior.\n\n"
            f"Available functions:\n{definitions}"
        )

    def parameter_system_message(
        self,
        name: str,
        parameter: str,
        position: int,
        total: int,
        extracted: dict[str, Any],
    ) -> str:
        """Build the Qwen system instruction for parameter extraction."""
        definition = json.dumps(
            self.get_function_definition(name), ensure_ascii=False
        )
        extracted_json = json.dumps(extracted, ensure_ascii=False)
        parameter_hint = ""
        normalized_parameter = parameter.lower()
        if "source" in normalized_parameter:
            parameter_hint = (
                "The target is the complete source text for the operation, "
                "not only the matched fragment. "
            )
        elif (
            "regex" in normalized_parameter
            or "pattern" in normalized_parameter
        ):
            parameter_hint = (
                "The target must be a valid regex for the user's matching "
                "rule. Translate the user's described rule to "
                "the appropriate pattern; do not output the natural-language "
                "description itself or list source matching examples. For a "
                "requested character category, the pattern must match each "
                "individual character and include case variants "
                "unless the request restricts case. Before answering, verify "
                "that the pattern matches every requested target and does not "
                "merely copy a matching fragment from the source text. "
            )
        elif "replacement" in normalized_parameter:
            parameter_hint = (
                "The target is the requested replacement text or symbol, not "
                "the source text or a matched value. Translate a named "
                "punctuation symbol to its literal character, rather than "
                "the word that names it. Do not copy a source fragment as the "
                "replacement unless the user explicitly requests that text. "
            )
        return (
            "Extract one input argument for the selected function. Return the "
            "argument supplied by the user, not the result of executing the "
            "function. Copy literal values that directly correspond to the "
            "parameter. When a parameter represents a rule, pattern, format, "
            "symbol, infer the value that implements the user's requested "
            "operation. "
            "Use request order only when direct literal values are ambiguous. "
            "Never use it instead of deriving a rule, pattern, format, or "
            "symbol. Continue the incomplete JSON in the assistant "
            "message and close the current value with a double quote. "
            f"{parameter_hint}"
            "Do not add an explanation.\n\n"
            f"Selected function:\n{definition}\n\n"
            f"Target parameter: {parameter}\n"
            f"Target position: {position} of {total}\n"
            f"Previously extracted arguments: {extracted_json}"
        )

    def encode_prompt(
        self,
        system_message: str,
        user_message: str,
        assistant_prefix: str = "",
    ) -> list[int]:
        """Encode a textual instruction and its constrained continuation."""
        prompt = (
            "<|im_start|>system\n"
            f"{system_message}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_message}<|im_end|>\n"
            "<|im_start|>assistant\n"
            f"{assistant_prefix}"
        )
        return self.llm.encode(prompt)[0].tolist()

    def decode_token(self, token_id: int) -> str:
        """Decode and cache one token's visible text."""
        if token_id not in self._token_cache:
            self._token_cache[token_id] = self.llm.decode([token_id])
        return self._token_cache[token_id]

    def select_name(self, user_prompt: str) -> str:
        """Generate one valid function name using a prefix mask."""
        ids = self.encode_prompt(
            self.selection_system_message(), user_prompt, assistant_prefix='"'
        )
        name = ""

        for _ in range(self.max_tokens_per_value):
            logits = self.llm.get_logits_from_input_ids(ids)
            for token_id in range(len(logits)):
                token = self.decode_token(token_id)
                candidate = name + token
                if '"' in candidate:
                    candidate_name = candidate.split('"', 1)[0]
                    if candidate_name not in self._function_names:
                        logits[token_id] = float("-inf")
                    continue
                if candidate not in self._function_prefixes:
                    logits[token_id] = float("-inf")

            next_token_id = logits.index(max(logits))
            token = self.decode_token(next_token_id)
            if not token and name in self._function_names:
                return name
            if '"' in token:
                return name + token.split('"', 1)[0]
            name += token
            matching_names = [
                function_name
                for function_name in self._function_names
                if function_name.startswith(name)
            ]
            if name in self._function_names and len(matching_names) == 1:
                return name
            ids.append(next_token_id)

        raise RuntimeError(
            "Function-name generation exceeded its token limit."
        )

    @staticmethod
    def is_number_prefix(value: str) -> bool:
        """Return whether text can still become a JSON number."""
        return re.fullmatch(
            r"-?(?:\d+(?:\.\d*)?|\.\d+)?(?:[eE][+-]?\d*)?", value
        ) is not None

    def is_valid_parameter_candidate(
        self,
        candidate: str,
        parameter_type: str,
        number_candidates: list[str],
        string_candidates: list[str],
    ) -> bool:
        """Keep only tokens compatible with the current value type."""
        if not candidate:
            return False
        if parameter_type == "string" and not string_candidates:
            structural_positions = [
                candidate.find(marker)
                for marker in ('"', "\n", "}", "<")
                if marker in candidate
            ]
            if structural_positions:
                return bool(candidate[:min(structural_positions)])
        has_closing_quote = '"' in candidate
        if has_closing_quote and (
            not candidate.endswith('"') or candidate.count('"') != 1
        ):
            return False
        value = candidate[:-1] if has_closing_quote else candidate
        if parameter_type == "string":
            if string_candidates:
                if has_closing_quote:
                    return value in string_candidates
                return any(
                    item.startswith(value) for item in string_candidates
                )
            return bool(value) or has_closing_quote
        if parameter_type == "number":
            if number_candidates:
                if has_closing_quote:
                    return value in number_candidates
                return any(
                    number.startswith(value) for number in number_candidates
                )
            if has_closing_quote:
                return self.is_complete_parameter_value(value, parameter_type)
            return self.is_number_prefix(value) and bool(value)
        raise ValueError(f"Unsupported parameter type: {parameter_type}")

    @staticmethod
    def is_complete_parameter_value(value: str, parameter_type: str) -> bool:
        """Return whether EOS may safely terminate the current value."""
        if parameter_type == "string":
            return bool(value.strip())
        if parameter_type == "number":
            try:
                float(value)
            except ValueError:
                return False
            return True
        return False

    @staticmethod
    def remaining_number_candidates(
        user_prompt: str,
        extracted: dict[str, Any],
    ) -> list[str]:
        """Return numeric literals from the request that remain unassigned."""
        candidates = re.findall(r"-?(?:\d+(?:\.\d*)?|\.\d+)", user_prompt)
        for extracted_value in extracted.values():
            if not isinstance(extracted_value, (int, float)):
                continue
            for index, candidate in enumerate(candidates):
                if float(candidate) == float(extracted_value):
                    candidates.pop(index)
                    break
        return candidates

    def extract_parameters(
        self, user_prompt: str, name: str
    ) -> dict[str, Any]:
        """Generate schema-ordered values in one Qwen JSON reply."""
        definition = self.get_function_definition(name)
        parameters = definition.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"Invalid parameters schema for {name}.")

        values: dict[str, Any] = {}
        total_parameters = len(parameters)

        for index, (parameter, schema) in enumerate(parameters.items()):
            if not isinstance(schema, dict) or not isinstance(
                schema.get("type"), str
            ):
                raise ValueError(f"Invalid schema for parameter {parameter}.")
            parameter_type = schema["type"]
            system_message = self.parameter_system_message(
                name,
                parameter,
                index + 1,
                total_parameters,
                values,
            )
            serialized_values = json.dumps(values, ensure_ascii=False)
            if values:
                arguments_prefix = (
                    serialized_values[:-1]
                    + f', {json.dumps(parameter)}: "'
                )
            else:
                arguments_prefix = f'{{{json.dumps(parameter)}: "'
            assistant_prefix = f"{arguments_prefix}"
            number_candidates = (
                self.remaining_number_candidates(user_prompt, values)
                if parameter_type == "number"
                else []
            )
            string_candidates: list[str] = []
            value = ""
            ids = self.encode_prompt(
                system_message,
                user_prompt,
                assistant_prefix,
            )

            for _ in range(self.max_tokens_per_value):
                logits = self.llm.get_logits_from_input_ids(ids)
                for token_id in range(len(logits)):
                    token = self.decode_token(token_id)
                    if not token:
                        logits[token_id] = float("-inf")
                        continue
                    if not self.is_valid_parameter_candidate(
                        value + token,
                        parameter_type,
                        number_candidates,
                        string_candidates,
                    ):
                        logits[token_id] = float("-inf")

                next_token_id = logits.index(max(logits))
                token = self.decode_token(next_token_id)
                terminators = ['"']
                if parameter_type == "string" and not string_candidates:
                    terminators.extend(["\n", "}", "<"])
                terminator_positions = [
                    token.find(marker)
                    for marker in terminators
                    if marker in token
                ]
                if terminator_positions:
                    value += token[:min(terminator_positions)]
                    break
                value += token
                if string_candidates:
                    matches = [
                        candidate
                        for candidate in string_candidates
                        if candidate.startswith(value)
                    ]
                    if value in string_candidates and len(matches) == 1:
                        break
                ids.append(next_token_id)
            else:
                raise RuntimeError(
                    "Parameter generation exceeded its token limit: "
                    f"{parameter}"
                )

            values[parameter] = (
                float(value) if parameter_type == "number" else value
            )

        return values

    def output(self, user_prompt: str) -> str:
        """Return one schema-shaped function-call JSON object."""
        name = self.select_name(user_prompt)
        parameters = self.extract_parameters(user_prompt, name)
        return json.dumps(
            {
                "prompt": user_prompt,
                "name": name,
                "parameters": parameters,
            },
            ensure_ascii=False,
            indent=4,
        )


def load_prompts(path: Path) -> list[str]:
    """Read prompt strings from the project input fixture."""
    try:
        with path.open("r", encoding="utf-8") as file:
            items = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Input file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in input file: {error.msg}") from error

    if not isinstance(items, list):
        raise ValueError("Input prompts must be a JSON array.")
    return [
        item["prompt"]
        for item in items
        if isinstance(item, dict) and isinstance(item.get("prompt"), str)
    ]


def parse_arguments() -> argparse.Namespace:
    """Parse the input, schema, and output paths accepted by the program."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate constrained function calls from user prompts."
        )
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to the JSON array that defines available functions.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to the JSON array containing prompt objects.",
    )
    parser.add_argument(
        "--output",
        "-output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path for the generated function-call JSON array.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate function calls using command-line supplied paths."""
    arguments = parse_arguments()
    start = time.time()
    function_caller = FunctionCalling(
        path_fn_definition=arguments.functions_definition
    )
    results = [
        json.loads(function_caller.output(prompt))
        for prompt in load_prompts(arguments.input)
    ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
        file.write("\n")
    print(f"Elapsed: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
