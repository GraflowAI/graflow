"""Factory for creating prompt manager instances."""

from __future__ import annotations

from typing import Any

from graflow.prompts.base import PromptManager


class PromptManagerFactory:
    """Factory for creating prompt manager instances.

    Supported backends:
        - "yaml": Local filesystem storage with YAML files (default)
        - "langfuse": Langfuse cloud/server integration (requires langfuse package)
    """

    @classmethod
    def create(
        cls,
        backend: str = "yaml",
        **kwargs: Any,
    ) -> PromptManager:
        """Create prompt manager instance.

        Args:
            backend: Backend type ("yaml" or "langfuse")
            **kwargs: Backend-specific configuration

        YAML backend kwargs:
            prompts_dir: Directory path (optional, defaults to GRAFLOW_PROMPTS_DIR
                        env var or "./prompts")
            cache_ttl: Default cache TTL in seconds (default: 300). Per-entry TTL
                      can be overridden via get_prompt(cache_ttl_seconds=...).
                      Set to 0 for no expiration.
            cache_maxsize: Maximum cached prompt entries (default: 1000).
                          Uses TLRUCache with per-entry TTL and LRU eviction.

        Langfuse backend kwargs:
            public_key: Langfuse public key (optional, from LANGFUSE_PUBLIC_KEY env)
            secret_key: Langfuse secret key (optional, from LANGFUSE_SECRET_KEY env)
            host: Langfuse host (optional, from LANGFUSE_HOST env)
            fetch_timeout_seconds: Timeout for fetching prompts (optional, uses SDK default if None)
            max_retries: Maximum retry attempts (optional, uses SDK default if None)

        Returns:
            PromptManager instance

        Raises:
            ValueError: If backend is unknown or unavailable

        Example:
            ```python
            # YAML backend (default)
            pm = PromptManagerFactory.create()
            pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

            # Langfuse backend
            pm = PromptManagerFactory.create("langfuse")
            ```
        """
        if backend == "yaml":
            from graflow.prompts.yaml_manager import YAMLPromptManager

            return YAMLPromptManager(**kwargs)

        elif backend == "langfuse":
            from graflow.prompts.langfuse_manager import LANGFUSE_AVAILABLE, LangfusePromptManager

            if not LANGFUSE_AVAILABLE:
                raise ValueError(
                    "Langfuse backend is not available. "
                    "Install it with: pip install langfuse or pip install graflow[tracing]"
                )
            return LangfusePromptManager(**kwargs)

        else:
            raise ValueError(
                f"Unknown backend: '{backend}'. Available: 'yaml' (always), 'langfuse' (requires langfuse package)"
            )
