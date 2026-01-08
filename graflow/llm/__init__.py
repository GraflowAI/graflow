"""LLM integration for Graflow.

This module provides LiteLLM integration for LLM completion API and Google ADK
integration for agent-based patterns (ReAct, Supervisor).

Key components:
- LLMClient: Thin wrapper around LiteLLM with tracing support
- setup_langfuse_for_litellm: Enable Langfuse tracing (loads from .env)
- LLMAgent: Base class for agent implementations
- AdkLLMAgent: Wrapper for Google ADK LlmAgent

Example:
    ```python
    from graflow.llm import LLMClient, setup_langfuse_for_litellm
    from graflow.core.decorators import task

    # Enable Langfuse tracing (optional, loads from .env)
    setup_langfuse_for_litellm()

    # Create LLM client
    llm_client = LLMClient(model="gpt-4o-mini", temperature=0.7)

    # Use in tasks
    @task(inject_llm_client=True)
    def generate_summary(llm: LLMClient, text: str) -> str:
        response = llm.completion(
            messages=[
                {"role": "system", "content": "Summarize the text."},
                {"role": "user", "content": text}
            ],
            generation_name="summary"
        )
        return extract_text(response)
    ```

Agent example:
    ```python
    from google.adk.agents import LlmAgent
    from graflow.llm import AdkLLMAgent
    from graflow.core.decorators import task

    # Create ADK agent
    adk_agent = LlmAgent(
        name="supervisor",
        model="gemini-2.5-flash",
        tools=[search_tool, calculator_tool]
    )
    agent = AdkLLMAgent(adk_agent, app_name=context.trace_id)

    # Register agent
    context.register_llm_agent("supervisor", agent)

    # Use in tasks
    @task(inject_llm_agent="supervisor")
    def run_analysis(agent: LLMAgent, query: str) -> str:
        result = agent.run(query)
        return result["output"]
    ```
"""

from graflow.llm.agents import LLMAgent
from graflow.llm.client import LLMClient, extract_text, make_message

__all__ = [
    "LLMAgent",
    "LLMClient",
    "extract_text",
    "make_message",
]

# Optional: Agent classes (only if dependencies available)
try:
    from graflow.llm.agents import AdkLLMAgent  # noqa: F401

    __all__.extend(["AdkLLMAgent"])
except ImportError:
    pass

# Optional: Pydantic AI agent (only if pydantic-ai is installed)
try:
    from graflow.llm.agents import (  # noqa: F401
        PydanticLLMAgent,
        create_pydantic_ai_agent_with_litellm,
    )

    __all__.extend(["PydanticLLMAgent", "create_pydantic_ai_agent_with_litellm"])
except ImportError:
    pass

# Optional: Serialization helpers
try:
    from graflow.llm.serialization import agent_to_yaml, yaml_to_agent  # noqa: F401

    __all__.extend(["agent_to_yaml", "yaml_to_agent"])
except ImportError:
    pass
