"""Configuration for GPT Newspaper"""

import os


class Config:
    """Configuration class for GPT Newspaper."""

    # API Keys
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # LLM Configuration
    DEFAULT_MODEL = os.getenv("GPT_NEWSPAPER_MODEL", "gpt-4o-mini")

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

        if not cls.OPENAI_API_KEY:
            print("⚠️  Warning: OPENAI_API_KEY not found.")
            print("Make sure your LLM provider API key is configured.")
            print("Supported providers: OpenAI, Anthropic, Cohere, etc.")
            print("See: https://docs.litellm.ai/docs/providers")

        return True

    @classmethod
    def display(cls):
        """Display current configuration."""
        print("=" * 80)
        print("⚙️  Configuration")
        print("=" * 80)
        print(f"LLM Model: {cls.DEFAULT_MODEL}")
        print(f"Layout: {cls.DEFAULT_LAYOUT}")
        print(f"Max Iterations: {cls.MAX_CRITIQUE_ITERATIONS}")
        print(f"Output Directory: {cls.OUTPUT_BASE_DIR}")
        print(f"Tavily API Key: {'✅ Set' if cls.TAVILY_API_KEY else '❌ Not Set'}")
        print(f"OpenAI API Key: {'✅ Set' if cls.OPENAI_API_KEY else '❌ Not Set'}")
        print("=" * 80)
        print()
