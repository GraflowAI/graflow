"""Langfuse-based prompt manager for cloud/server storage."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langfuse import Langfuse

# Try to import langfuse
try:
    from langfuse import Langfuse

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

from graflow.prompts.base import PromptManager
from graflow.prompts.models import ChatPrompt, PromptVersion, TextPrompt


class LangfusePromptManager(PromptManager):
    """Langfuse-based prompt manager with cloud/server storage."""

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        fetch_timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """Initialize Langfuse prompt manager.

        Args:
            public_key: Langfuse public key (from LANGFUSE_PUBLIC_KEY env if None)
            secret_key: Langfuse secret key (from LANGFUSE_SECRET_KEY env if None)
            host: Langfuse host (from LANGFUSE_HOST env if None)
            fetch_timeout_seconds: Timeout in seconds for fetching prompts from Langfuse server
                                  None = use Langfuse SDK default
            max_retries: Maximum number of retry attempts for failed fetches
                        None = use Langfuse SDK default

        Raises:
            ImportError: If langfuse package is not installed

        Note:
            Langfuse SDK handles caching internally via cache_ttl_seconds parameter.
            No custom cache implementation needed in this manager.

            fetch_timeout_seconds and max_retries are applied to all get_prompt() calls.
        """
        if not LANGFUSE_AVAILABLE:
            raise ImportError(
                "Langfuse package is not installed. "
                "Install it with: pip install langfuse or pip install graflow[tracing]"
            )

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self.max_retries = max_retries

    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> PromptVersion:
        """Get prompt from Langfuse.

        Args:
            name: Prompt name
            version: Numeric version
            label: Version label
            cache_ttl_seconds: Cache TTL in seconds

        Returns:
            PromptVersion object (TextPrompt or ChatPrompt)

        Raises:
            ValueError: If both version and label specified
        """
        # Validate: cannot specify both label and version
        if version is not None and label is not None:
            raise ValueError(
                "Cannot specify both 'version' and 'label'. "
                "Use label for environment-based access or version for direct access."
            )

        # Map cache_ttl_seconds to Langfuse's cache_ttl_seconds
        if cache_ttl_seconds == -1:
            cache_ttl_seconds = 0  # No cache
        elif cache_ttl_seconds == 0:
            # Langfuse doesn't support infinite cache, use default
            cache_ttl_seconds = 60 * 60 * 6  # 6 hours as a practical long cache

        # Fetch from Langfuse with SDK's built-in caching
        # Returns Union[TextPromptClient, ChatPromptClient]
        langfuse_prompt = self._client.get_prompt(
            name,
            version=version,
            label=label,
            cache_ttl_seconds=cache_ttl_seconds,
            max_retries=self.max_retries,
            fetch_timeout_seconds=self.fetch_timeout_seconds,
        )

        # Convert to PromptVersion (TextPrompt or ChatPrompt)
        return self._convert_langfuse_prompt(langfuse_prompt)

    def _convert_langfuse_prompt(self, langfuse_prompt: Any) -> PromptVersion:
        """Convert Langfuse prompt object to TextPrompt or ChatPrompt.

        Args:
            langfuse_prompt: Langfuse prompt object (TextPromptClient or ChatPromptClient)

        Returns:
            TextPrompt or ChatPrompt instance

        Note:
            Langfuse SDK returns different types:
            - TextPromptClient: has .prompt (str) and .type == 'text'
            - ChatPromptClient: has .prompt (List[ChatMessageDict]) and .type == 'chat'

            Both have: .name, .version, .labels, .config
        """
        # Detect type from the prompt object
        # Langfuse SDK provides 'type' attribute
        if hasattr(langfuse_prompt, "type"):
            prompt_type = langfuse_prompt.type  # 'text' or 'chat'
        else:
            # Fallback: infer from content type
            prompt_type = "chat" if isinstance(langfuse_prompt.prompt, list) else "text"

        # Extract content (already in the right format)
        content = langfuse_prompt.prompt

        # Extract version info
        version_num = getattr(langfuse_prompt, "version", None)

        # Extract label (Langfuse stores as list, take first)
        labels = getattr(langfuse_prompt, "labels", [])
        label = labels[0] if labels else None

        # Extract metadata
        metadata = getattr(langfuse_prompt, "config", {})

        # Return appropriate type
        if prompt_type == "chat":
            assert isinstance(content, list)
            return ChatPrompt(
                name=langfuse_prompt.name,
                content=content,
                version=version_num,
                label=label,
                created_at=None,
                metadata=metadata,
            )
        else:
            assert isinstance(content, str)
            return TextPrompt(
                name=langfuse_prompt.name,
                content=content,
                version=version_num,
                label=label,
                created_at=None,
                metadata=metadata,
            )
