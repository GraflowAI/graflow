"""Custom exceptions for prompt management."""


class PromptError(Exception):
    """Base exception for prompt management errors."""

    pass


class PromptNotFoundError(PromptError):
    """Raised when prompt is not found."""

    pass


class PromptVersionNotFoundError(PromptError):
    """Raised when prompt version/label is not found."""

    pass


class PromptCompilationError(PromptError):
    """Raised when template rendering fails."""

    pass


class PromptConfigurationError(PromptError):
    """Raised when prompt manager configuration is invalid or duplicate prompts detected."""

    pass


class PromptTypeError(PromptError):
    """Raised when prompt type does not match expected type (text vs chat)."""

    pass
