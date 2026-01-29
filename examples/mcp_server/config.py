"""Configuration for Company Intelligence MCP Server."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Then load example-specific .env (override project settings if needed)
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path, override=True)


class Config:
    """Configuration management for the MCP server."""

    # LLM Models
    DEFAULT_MODEL = os.getenv("GRAFLOW_LLM_MODEL", "gpt-4o-mini")
    WRITER_MODEL = os.getenv("WRITER_MODEL", "gpt-4o-mini")
    CRITIQUE_MODEL = os.getenv("CRITIQUE_MODEL", "gpt-4o-mini")

    # API Keys
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Langfuse Tracing
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    ENABLE_TRACING = os.getenv("ENABLE_TRACING", "true").lower() == "true"

    # Workflow Configuration
    MAX_CRITIQUE_ITERATIONS = int(os.getenv("MAX_CRITIQUE_ITERATIONS", "3"))
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))

    # MCP Server Configuration
    MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "9100"))

    @classmethod
    def display(cls) -> None:
        """Display current configuration."""
        print("=" * 50)
        print("Company Intelligence MCP Server Configuration")
        print("=" * 50)
        print(f"DEFAULT_MODEL: {cls.DEFAULT_MODEL}")
        print(f"WRITER_MODEL: {cls.WRITER_MODEL}")
        print(f"CRITIQUE_MODEL: {cls.CRITIQUE_MODEL}")
        print(f"TAVILY_API_KEY: {'***' if cls.TAVILY_API_KEY else '(not set)'}")
        print(f"OPENAI_API_KEY: {'***' if cls.OPENAI_API_KEY else '(not set)'}")
        print(f"LANGFUSE_HOST: {cls.LANGFUSE_HOST}")
        print(f"ENABLE_TRACING: {cls.ENABLE_TRACING}")
        print(f"MAX_CRITIQUE_ITERATIONS: {cls.MAX_CRITIQUE_ITERATIONS}")
        print(f"MAX_SEARCH_RESULTS: {cls.MAX_SEARCH_RESULTS}")
        print(f"MCP_SERVER: {cls.MCP_SERVER_HOST}:{cls.MCP_SERVER_PORT}")
        print("=" * 50)

    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration."""
        errors = []

        if not cls.TAVILY_API_KEY:
            errors.append("TAVILY_API_KEY is required for web search")

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required for LLM calls")

        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    @classmethod
    def get_langfuse_config(cls) -> dict[str, Any] | None:
        """Get Langfuse configuration if enabled."""
        if not cls.ENABLE_TRACING:
            return None

        if not cls.LANGFUSE_PUBLIC_KEY or not cls.LANGFUSE_SECRET_KEY:
            print("Warning: Langfuse keys not set, tracing disabled")
            return None

        return {
            "public_key": cls.LANGFUSE_PUBLIC_KEY,
            "secret_key": cls.LANGFUSE_SECRET_KEY,
            "host": cls.LANGFUSE_HOST,
        }
