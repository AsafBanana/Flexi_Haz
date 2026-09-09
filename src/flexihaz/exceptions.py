class FlexiHazError(Exception):
    """Base exception for flexihaz."""


class NotFittedError(FlexiHazError):
    """Raised when prediction is requested before fitting."""


class InputValidationError(FlexiHazError, ValueError):
    """Raised when user input is invalid."""
