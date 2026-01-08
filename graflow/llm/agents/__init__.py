"""LLM agent implementations for Graflow."""

from graflow.llm.agents.base import LLMAgent

__all__ = ["LLMAgent"]

# Optional: ADK agent (only if google-adk is installed)
try:
    from graflow.llm.agents.adk_agent import AdkLLMAgent  # noqa: F401

    __all__.append("AdkLLMAgent")
except ImportError:
    pass

# Optional: Pydantic AI agent (only if pydantic-ai is installed)
try:
    from graflow.llm.agents.pydantic_agent import (  # noqa: F401
        PydanticLLMAgent,
        create_pydantic_ai_agent_with_litellm,
    )

    __all__.extend(["PydanticLLMAgent", "create_pydantic_ai_agent_with_litellm"])
except ImportError:
    pass
