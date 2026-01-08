"""Utility wrapper for integrating LiteLLM with graflow tasks.

This helper keeps LiteLLM as an optional dependency while providing a small,
focused API that is convenient to call from graflow tasks.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Mapping


class LiteLLMClient:
    """Thin convenience wrapper around :mod:`litellm`.

    The wrapper keeps a shared model configuration and exposes helper
    methods that return the plain response object as well as utilities
    to extract text content.
    """

    def __init__(self, model: str, **default_params: Any) -> None:
        """Create a new LiteLLM client.

        Args:
            model: Model identifier understood by LiteLLM (e.g. ``"gpt-4o-mini"``).
            **default_params: Default keyword arguments forwarded to
                :func:`litellm.completion`.

        Raises:
            RuntimeError: If :mod:`litellm` is not installed.
        """
        try:
            self._litellm = importlib.import_module("litellm")
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError("liteLLM is not installed. Install it with 'pip install litellm'.") from exc

        self.model = model
        self.default_params = default_params

    # ------------------------------------------------------------------ #
    # Chat completion helpers
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: Iterable[Mapping[str, Any]],
        **params: Any,
    ) -> Any:
        """Run a chat completion with the configured model.

        Args:
            messages: Iterable of message dictionaries ``{"role": ..., "content": ...}``.
            **params: Extra keyword arguments forwarded to
                :func:`litellm.completion`, overriding defaults.

        Returns:
            The raw LiteLLM response (``litellm.utils.ModelResponse``).
        """
        kwargs = {**self.default_params, **params}
        return self._litellm.completion(model=self.model, messages=list(messages), **kwargs)

    def chat_text(
        self,
        messages: Iterable[Mapping[str, Any]],
        **params: Any,
    ) -> str:
        """Shortcut that returns the first text choice."""
        response = self.chat(messages, **params)
        return extract_text(response)


def extract_text(response: Any) -> str:
    """Extract ``message.content`` from a LiteLLM completion response.

    LiteLLM returns a ``ModelResponse`` object, but some tools cast it
    to a dict, so we support both.
    """
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")

    if not choices:
        return ""

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, Mapping):
        message = choice.get("message", {})

    content = getattr(message, "content", None)
    if content is None and isinstance(message, Mapping):
        content = message.get("content")

    return content or ""


def make_message(role: str, content: str) -> Dict[str, str]:
    """Convenience helper to create OpenAI-style chat messages."""
    return {"role": role, "content": content}
