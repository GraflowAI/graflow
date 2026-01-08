"""Utility for safe dotenv loading across the codebase."""

from __future__ import annotations

# Optional dotenv import
try:
    from dotenv import load_dotenv as _load_dotenv  # type: ignore[import-not-found]

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    _load_dotenv = None  # type: ignore


def load_env() -> bool:
    """Load environment variables from .env file if dotenv is available.

    Returns:
        True if dotenv was loaded successfully, False otherwise

    Example:
        ```python
        from graflow.utils.dotenv import load_env

        # Safe to call even if python-dotenv is not installed
        load_env()

        # Get environment variables as usual
        import os
        model = os.getenv("GRAFLOW_LLM_MODEL", "gpt-5-mini")
        ```
    """
    if DOTENV_AVAILABLE and _load_dotenv:
        _load_dotenv()
        return True
    return False
