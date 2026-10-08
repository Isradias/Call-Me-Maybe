*This project has been created as part of the 42 curriculum by icaldas-.*

# Call Me Maybe

## Description

Call Me Maybe is a function-calling tool built around `Qwen/Qwen3-0.6B`. It reads a catalogue of function definitions and natural-language prompts, then produces one schema-valid JSON function call for each prompt. It returns the function to invoke and its typed arguments; it does not execute the selected function.

The project focuses on constrained decoding. At every generation step, it considers only model tokens that can extend the current output into a valid JSON object following the supplied function schema. This prevents malformed JSON, unknown function names, unexpected parameters, missing parameters, and top-level type mismatches.

## Repository layout

```text
src/                    Application package and command-line entry point
llm_sdk/                Provided wrapper for the local language model
data/input/             Example function definitions and prompt inputs
Makefile                Common development commands
pyproject.toml          Project metadata and dependencies
uv.lock                 Locked dependency resolution
```

Generated results are written to `data/output/` by default. That directory is ignored by Git and is not part of the submission.

## Instructions

### Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- Internet access on the first run so the model files can be downloaded

Install the locked dependencies:

```bash
uv sync
```

Run the program with the bundled demonstration inputs:

```bash
uv run python -m src
```

Or provide explicit input and output paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

The command prints an error to standard error and returns a non-zero exit status when an input document is missing, malformed, or incompatible with the expected schema.

### Makefile

```bash
make install             # Run uv sync
make run                 # Run the application
make debug               # Run through Python's debugger
make lint                # Run flake8 and the required mypy checks
make lint-strict         # Run stricter mypy checks
make test                # Run the unit tests
make clean               # Remove caches and generated output
```

## Input and output format

`data/input/functions_definition.json` is an array of functions. Each function has a unique `name`, a `description`, a `parameters` object, and a `returns` object. Parameter types supported by the decoder are `string`, `number`, `integer`, `boolean`, `null`, `object`, and `array`.

`data/input/function_calling_tests.json` is an array of objects containing a non-empty `prompt` string. The evaluator may replace both input files, so the implementation does not hard-code the bundled catalogue or prompts.

The generated output is a JSON array. Every item contains exactly:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

## Constrained-decoding algorithm

1. The application loads and validates the function catalogue and prompt list with Pydantic models.
2. It builds an instruction containing the available functions and the current user request.
3. The SDK tokenizes that instruction and returns logits for the next token.
4. For each candidate token, the decoder appends its visible text to the generated JSON prefix and checks whether the prefix can still become a complete function-call object.
5. The prefix parser enforces the fixed object shape (`name` and `parameters`), limits `name` to a declared function, allows each declared parameter once, requires every parameter, and validates the top-level JSON type of every value.
6. Tokens that fail those checks are discarded. The highest-logit remaining token is appended, and the process repeats until the JSON object is complete.
7. The completed JSON is parsed and validated again before it is added to the output. Results are written atomically only after every prompt succeeds.

The decoder operates at token level, rather than trusting a prompt to make the model produce valid JSON on its own. The second validation step is defensive: it protects the output boundary even if the model wrapper or vocabulary behaves unexpectedly.

## Design decisions

- **Pydantic at the boundaries:** input files and generated results are represented by explicit models, with unknown fields rejected.
- **Public SDK interface only:** the application uses the SDK's documented encoding, decoding, logits, and vocabulary-path methods. A protocol describes that interface so the generation logic can be unit tested without loading the model.
- **Schema-driven validation:** function names and parameter names are derived from the supplied JSON catalogue, not from a hard-coded list.
- **Greedy valid-token selection:** among tokens that preserve a valid prefix, the decoder selects the highest-logit candidate for deterministic behavior.
- **Atomic output writes:** a temporary file is replaced only after the full batch has been processed, avoiding partially written result files.
- **Clear failures:** file, JSON, validation, model, and generation failures are converted into user-facing error messages instead of uncaught tracebacks.

## Performance and reliability

The model is initialized once per program run, and its vocabulary is loaded once before processing all prompts. Decoded token text is cached within each generated call.

Generation checks the vocabulary at each output step, so its cost grows with both the number of generated tokens and the vocabulary size. This is a deliberate trade-off for reliability: no selected token is accepted unless it preserves a possible schema-valid JSON prefix. The project does not claim a fixed accuracy or timing value because both depend on the model files, hardware, function catalogue, and evaluation prompts. Run the bundled inputs on the target machine to measure them.

## Challenges faced

- JSON must remain valid even when a tokenizer emits multi-character tokens. The prefix parser therefore validates the complete candidate token text, not individual characters alone.
- A syntactically valid JSON object is not sufficient. The decoder also needs to reject unknown functions, repeated or extra parameters, missing required parameters, and incompatible JSON types.
- Model generation can fail for environmental reasons, including unavailable model files or invalid vocabularies. These cases are surfaced as controlled errors.
- Output should not be left in a partly updated state if a later prompt fails. Atomic writes solve that failure mode.

## Testing strategy

The unit tests cover the schema-prefix validator and the post-generation validator. In particular, they verify that valid calls complete successfully and that extra, missing, or incorrectly typed parameters are rejected. Run them with:

```bash
make test
```

Before submission, also run:

```bash
make lint
uv run python -m src
```

Then inspect the generated JSON to confirm that every function name and parameter value matches the supplied schema.

## Resources

- [Qwen documentation](https://qwen.readthedocs.io/)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

### Use of AI

AI tools were used as a collaborative aid to review repository structure, explain the project requirements, and draft documentation. All generated suggestions were reviewed, adapted to the implemented code, and tested by the author before inclusion.
