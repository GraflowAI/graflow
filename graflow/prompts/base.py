"""Abstract base class for prompt management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from graflow.prompts.exceptions import PromptTypeError
from graflow.prompts.models import ChatPrompt, PromptVersion, TextPrompt


class PromptManager(ABC):
    """Abstract base class for prompt management backends."""

    @abstractmethod
    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> PromptVersion:
        """Get prompt by name with optional version/label.

        Args:
            name: Prompt name (can include folder path, e.g., "customer/greeting")
            version: Numeric version (1, 2, 3...)
            label: Version label ("production", "staging", "latest")
            cache_ttl_seconds: Cache TTL in seconds

        Returns:
            PromptVersion object (TextPrompt or ChatPrompt)

        Raises:
            ValueError: If both version and label specified
            PromptNotFoundError: If prompt not found
            PromptVersionNotFoundError: If version/label not found
        """
        pass

    def get_text_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> TextPrompt:
        """Get text prompt by name with optional version/label.

        Args:
            name: Prompt name (can include folder path, e.g., "customer/greeting")
            version: Numeric version (1, 2, 3...)
            label: Version label ("production", "staging", "latest")
            cache_ttl_seconds: Cache TTL in seconds

        Returns:
            TextPrompt object with render() -> str method

        Raises:
            ValueError: If both version and label specified
            PromptNotFoundError: If prompt not found
            PromptVersionNotFoundError: If version/label not found
            PromptTypeError: If prompt is not a text type
        """
        prompt = self.get_prompt(name, version=version, label=label, cache_ttl_seconds=cache_ttl_seconds)

        if not isinstance(prompt, TextPrompt):
            raise PromptTypeError(
                f"Prompt '{name}' is a chat prompt, not a text prompt. Use get_chat_prompt() instead."
            )

        return prompt

    def get_chat_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> ChatPrompt:
        """Get chat prompt by name with optional version/label.

        Args:
            name: Prompt name (can include folder path, e.g., "customer/assistant")
            version: Numeric version (1, 2, 3...)
            label: Version label ("production", "staging", "latest")
            cache_ttl_seconds: Cache TTL in seconds

        Returns:
            ChatPrompt object with render() -> List[Dict[str, Any]] method

        Raises:
            ValueError: If both version and label specified
            PromptNotFoundError: If prompt not found
            PromptVersionNotFoundError: If version/label not found
            PromptTypeError: If prompt is not a chat type
        """
        prompt = self.get_prompt(name, version=version, label=label, cache_ttl_seconds=cache_ttl_seconds)

        if not isinstance(prompt, ChatPrompt):
            raise PromptTypeError(
                f"Prompt '{name}' is a text prompt, not a chat prompt. Use get_text_prompt() instead."
            )

        return prompt
