"""Application-specific exceptions with user-facing messages."""


class InputError(Exception):
    """Raised when an input or output document cannot be handled safely."""


class GenerationError(Exception):
    """Raised when constrained model generation cannot complete."""
