"""Configuration for GPT Newspaper."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

def _load_model_params(raw_value: str | None) -> Dict[str, Any]:
    """Parse model parameter configuration from JSON."""
    if not raw_value:
        return {}

    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        print("⚠️  Invalid GRAFLOW_MODEL_PARAMS; expected JSON object. Ignoring.")
        return {}

    if not isinstance(parsed, dict):
        print("⚠️  GRAFLOW_MODEL_PARAMS must be a JSON object. Ignoring.")
        return {}

    # Ensure keys are strings for LiteLLM kwargs.
    return {str(key): value for key, value in parsed.items()}


class Config:
    """Configuration class for GPT Newspaper."""

    # API Keys
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # LLM Configuration
    # Uses GRAFLOW_LLM_MODEL for Graflow's LLMClient integration
    DEFAULT_MODEL = os.getenv("GRAFLOW_LLM_MODEL", "gpt-4o-mini")
    DEFAULT_MODEL_PARAMS: Dict[str, Any] = _load_model_params(
        os.getenv("GRAFLOW_MODEL_PARAMS")
    )

    # Workflow Configuration
    MAX_CRITIQUE_ITERATIONS = int(os.getenv("MAX_CRITIQUE_ITERATIONS", "5"))
    DEFAULT_LAYOUT = os.getenv("NEWSPAPER_LAYOUT", "layout_1.html")

    # Output Configuration
    OUTPUT_BASE_DIR = os.getenv("OUTPUT_DIR", "outputs")

    @classmethod
    def validate(cls) -> bool:
        """
        Validate required configuration.

        Returns:
            True if configuration is valid
        """
        if not cls.TAVILY_API_KEY:
            print("❌ Error: TAVILY_API_KEY environment variable is required")
            print("Get your API key at: https://tavily.com/")
            return False

        if not cls.OPENAI_API_KEY and "api_base" not in cls.DEFAULT_MODEL_PARAMS:
            print("⚠️  Warning: OPENAI_API_KEY not found.")
            print("Make sure your LLM provider API key is configured.")
            print("Supported providers: OpenAI, Anthropic, Cohere, Ollama, etc.")
            print("See: https://docs.litellm.ai/docs/providers")

        return True

    @classmethod
    def display(cls) -> None:
        """Display current configuration."""
        print("=" * 80)
        print("⚙️  Configuration")
        print("=" * 80)
        print(f"LLM Model: {cls.DEFAULT_MODEL} (via GRAFLOW_LLM_MODEL)")
        if cls.DEFAULT_MODEL_PARAMS:
            print(f"LLM Params: {cls.DEFAULT_MODEL_PARAMS}")
        print(f"Layout: {cls.DEFAULT_LAYOUT}")
        print(f"Max Iterations: {cls.MAX_CRITIQUE_ITERATIONS}")
        print(f"Output Directory: {cls.OUTPUT_BASE_DIR}")
        print(f"Tavily API Key: {'✅ Set' if cls.TAVILY_API_KEY else '❌ Not Set'}")
        print(f"OpenAI API Key: {'✅ Set' if cls.OPENAI_API_KEY else '❌ Not Set'}")
        print("=" * 80)
        print()
