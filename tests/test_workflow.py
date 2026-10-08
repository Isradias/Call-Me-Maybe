"""Tests for final generated-call validation."""

import unittest

from src.errors import GenerationError
from src.models import FunctionDefinitions
from src.workflow import validate_call


class WorkflowTests(unittest.TestCase):
    """Verify the defensive validation after constrained generation."""

    def setUp(self) -> None:
        """Build one function schema."""
        self.functions = FunctionDefinitions.model_validate([
            {
                "name": "fn_greet",
                "description": "Greet a person.",
                "parameters": {"name": {"type": "string"}},
                "returns": {"type": "string"},
            }
        ])

    def test_valid_call_is_returned(self) -> None:
        """A matching generated document becomes an output model."""
        result = validate_call(
            '{"name":"fn_greet","parameters":{"name":"Ada"}}',
            "Greet Ada",
            self.functions,
        )
        self.assertEqual(result.parameters, {"name": "Ada"})

    def test_number_parameters_are_normalized_to_float(self) -> None:
        """A schema ``number`` is represented as a float in output."""
        functions = FunctionDefinitions.model_validate([
            {
                "name": "fn_add",
                "description": "Add two values.",
                "parameters": {"value": {"type": "number"}},
                "returns": {"type": "number"},
            }
        ])
        result = validate_call(
            '{"name":"fn_add","parameters":{"value":2}}',
            "Add 2",
            functions,
        )
        self.assertEqual(result.parameters, {"value": 2.0})

    def test_missing_parameter_fails(self) -> None:
        """A result missing a required parameter is rejected."""
        with self.assertRaises(GenerationError):
            validate_call('{"name":"fn_greet","parameters":{}}', "Greet Ada", self.functions)
