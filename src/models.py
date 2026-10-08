"""Pydantic models for input definitions and generated function calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator


SUPPORTED_TYPES = frozenset(
    {"string", "number", "integer", "boolean", "null", "object", "array"}
)


class ValueDefinition(BaseModel):
    """Describe the top-level JSON type of a value."""

    model_config = ConfigDict(extra="forbid")
    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        """Reject types unsupported by the constrained decoder."""
        if value not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported value type: {value}")
        return value


class FunctionDefinition(BaseModel):
    """Represent one callable function supplied by the input schema."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ValueDefinition]
    returns: ValueDefinition

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
        cls, value: dict[str, ValueDefinition]
    ) -> dict[str, ValueDefinition]:
        """Require non-empty parameter names."""
        if any(not name.strip() for name in value):
            raise ValueError("Parameter names cannot be blank.")
        return value


class FunctionDefinitions(RootModel[list[FunctionDefinition]]):
    """Represent the complete, unique set of callable functions."""

    @field_validator("root")
    @classmethod
    def validate_names(cls, value: list[FunctionDefinition]) -> list[FunctionDefinition]:
        """Require at least one uniquely named function."""
        if not value:
            raise ValueError("At least one function is required.")
        names = [function.name for function in value]
        if len(names) != len(set(names)):
            raise ValueError("Function names must be unique.")
        return value

    def by_name(self, name: str) -> FunctionDefinition:
        """Return a known function or raise ``KeyError``."""
        for function in self.root:
            if function.name == name:
                return function
        raise KeyError(name)


class PromptDefinition(BaseModel):
    """Represent one natural-language request."""

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject whitespace-only requests without changing original text."""
        if not value.strip():
            raise ValueError("Prompt cannot be blank.")
        return value


class PromptDefinitions(RootModel[list[PromptDefinition]]):
    """Represent the prompt input array."""


class FunctionCallResult(BaseModel):
    """Represent one schema-validated result written to the output array."""

    model_config = ConfigDict(extra="forbid")
    prompt: str
    name: str
    parameters: dict[str, Any]
