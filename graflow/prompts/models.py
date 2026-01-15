"""Data models for prompt management."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from graflow.prompts.exceptions import PromptCompilationError


class PromptVersion:
    """Base class for prompt versions."""

    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        label: Optional[str] = None,
        created_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a prompt version.

        Args:
            name: Prompt name
            version: Numeric version ID
            label: Version label
            created_at: ISO timestamp
            metadata: Additional metadata
        """
        self.name = name
        self.version = version
        self.label = label
        self.created_at = created_at
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name!r}, version={self.version}, label={self.label!r})"


class TextPrompt(PromptVersion):
    """Text prompt with string content."""

    def __init__(
        self,
        name: str,
        content: str,
        version: Optional[int] = None,
        label: Optional[str] = None,
        created_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a text prompt.

        Args:
            name: Prompt name
            content: Template string
            version: Numeric version ID
            label: Version label
            created_at: ISO timestamp
            metadata: Additional metadata
        """
        super().__init__(name, version, label, created_at, metadata)
        self.content = content

    def render(self, **variables: Any) -> str:
        """Render template with variables using Jinja2 StrictUndefined mode.

        Args:
            **variables: Template variables to substitute

        Returns:
            Compiled prompt string

        Raises:
            PromptCompilationError: If template rendering fails (missing variables, syntax errors)

        Example:
            ```python
            prompt = pm.get_text_prompt("greeting")
            result = prompt.render(name="Alice", product="Graflow")
            # result: "Hello Alice, welcome to Graflow!"
            ```
        """
        from jinja2 import Environment, StrictUndefined, TemplateError

        env = Environment(undefined=StrictUndefined)

        try:
            template = env.from_string(self.content)
            return template.render(**variables)
        except TemplateError as e:
            raise PromptCompilationError(f"Failed to compile prompt '{self.name}': {e}") from e


class ChatPrompt(PromptVersion):
    """Chat prompt with message list content."""

    def __init__(
        self,
        name: str,
        content: List[Dict[str, Any]],
        version: Optional[int] = None,
        label: Optional[str] = None,
        created_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a chat prompt.

        Args:
            name: Prompt name
            content: List of message dicts with role and content
            version: Numeric version ID
            label: Version label
            created_at: ISO timestamp
            metadata: Additional metadata
        """
        super().__init__(name, version, label, created_at, metadata)
        self.content = content

    def render(self, **variables: Any) -> List[Dict[str, Any]]:
        """Render template with variables using Jinja2 StrictUndefined mode.

        Args:
            **variables: Template variables to substitute

        Returns:
            Compiled message list

        Raises:
            PromptCompilationError: If template rendering fails (missing variables, syntax errors)

        Example:
            ```python
            prompt = pm.get_chat_prompt("interview")
            messages = prompt.render(domain="AI", topic="transformers")
            # messages: [
            #   {'role': 'system', 'content': 'You are an expert in AI.'},
            #   {'role': 'user', 'content': 'Interview me about transformers.'}
            # ]
            ```
        """
        from jinja2 import Environment, StrictUndefined, TemplateError

        env = Environment(undefined=StrictUndefined)

        try:
            compiled_messages: List[Dict[str, Any]] = []
            for msg in self.content:
                # Only render 'content' field if it's a string
                msg_content = msg.get("content")
                if isinstance(msg_content, str):
                    template = env.from_string(msg_content)
                    content: Any = template.render(**variables)
                else:
                    content = msg_content

                # Preserve all fields (role, content, tool_calls, etc.)
                compiled_msg: Dict[str, Any] = {**msg, "content": content}
                compiled_messages.append(compiled_msg)
            return compiled_messages
        except TemplateError as e:
            raise PromptCompilationError(f"Failed to compile prompt '{self.name}': {e}") from e
