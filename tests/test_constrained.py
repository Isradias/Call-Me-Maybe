"""Unit tests for schema-prefix validation without loading the LLM."""

import unittest

from src.constrained import validate_prefix
from src.models import FunctionDefinitions


class ConstrainedDecoderTests(unittest.TestCase):
    """Verify JSON and function-schema restrictions."""

    def setUp(self) -> None:
        """Build schemas independent from the repository fixture."""
        self.functions = FunctionDefinitions.model_validate([
            {
                "name": "fn_add",
                "description": "Add two values.",
                "parameters": {
                    "left": {"type": "number"},
                    "right": {"type": "number"},
                },
                "returns": {"type": "number"},
            },
            {
                "name": "fn_toggle",
                "description": "Toggle a value.",
                "parameters": {"enabled": {"type": "boolean"}},
                "returns": {"type": "boolean"},
            },
        ])

    def test_valid_schema_call_is_complete(self) -> None:
        """A declared name, keys, and types form a complete prefix."""
        status = validate_prefix(
            '{"name":"fn_add","parameters":{"left":2,"right":3}}',
            self.functions,
        )
        self.assertTrue(status.complete)

    def test_extra_parameter_is_rejected(self) -> None:
        """The selected function cannot receive undocumented keys."""
        status = validate_prefix(
            '{"name":"fn_toggle","parameters":{"enabled":true,"other":1',
            self.functions,
        )
        self.assertFalse(status.possible)

    def test_wrong_type_is_rejected(self) -> None:
        """A boolean parameter cannot be generated as a JSON string."""
        status = validate_prefix(
            '{"name":"fn_toggle","parameters":{"enabled":"true"',
            self.functions,
        )
        self.assertFalse(status.possible)
