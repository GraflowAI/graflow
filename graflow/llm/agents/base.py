"""Base class for LLM agents in Graflow."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List

from graflow.llm.agents.types import AgentResult


class LLMAgent(ABC):
    """Abstract base class for LLM agents.

    LLM agents encapsulate complex LLM interaction patterns such as:
    - ReAct (Reasoning + Acting) loops
    - Supervisor agents with hierarchical sub-agents
    - Tool-using agents
    - Multi-step planning agents

    Implementations should wrap existing agent frameworks (e.g., Google ADK)
    and provide a consistent interface for Graflow tasks.

    Example:
        ```python
        from graflow.core.decorators import task
        from graflow.llm.agents import LLMAgent

        @task(inject_llm_agent="supervisor")
        def run_supervisor(agent: LLMAgent, query: str) -> str:
            result = agent.run(query)
            return result["output"]
        ```
    """

    @abstractmethod
    def run(self, input_text: str, **kwargs: Any) -> AgentResult:
        """Run the agent synchronously with the given input.

        Args:
            input_text:
                Input text or query for the agent.
            **kwargs:
                Additional parameters specific to the concrete agent
                implementation (e.g. user_id, trace_id, session_id).

        Returns:
            AgentResult:
                A structured result object containing:

                - output:
                    Final agent output.
                    Either a free-form string or a structured Pydantic BaseModel
                    if the agent is configured with an output schema.
                - steps:
                    Execution trace emitted by the agent.
                - metadata:
                    Additional metadata such as model name, usage, or event counts.

        Example:
            ```python
            result = agent.run("What is the weather in Tokyo?")
            print(result["output"])
            ```
        """
        pass

    async def run_async(self, input_text: str, **kwargs: Any) -> AsyncIterator[Any]:
        """Run the agent asynchronously with streaming events.

        This method provides async execution with event streaming, mirroring
        the Google ADK Runner.run_async() API. Events can include partial
        responses (partial=True) for streaming and final responses.

        Args:
            input_text: Input text/query for the agent
            **kwargs: Additional parameters specific to the agent implementation

        Yields:
            Agent execution events. Event format depends on implementation.
            Typically includes events with:
            - partial=True: Intermediate streaming chunks
            - is_final_response(): Final result

        Raises:
            NotImplementedError: If async execution is not supported

        Example:
            ```python
            async for event in agent.run_async("Tell me a story"):
                if hasattr(event, 'partial') and event.partial:
                    print(event.content, end="", flush=True)
                elif event.is_final_response():
                    print(f"\\nFinal: {event.content}")
            ```

        Note:
            Default implementation raises NotImplementedError. Implementations
            should override this method if async execution is supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support async execution. Use run() for synchronous execution."
        )
        # Make this an async generator to satisfy type hints
        yield  # type: ignore[misc]

    @property
    @abstractmethod
    def name(self) -> str:
        """Get agent name/identifier."""
        pass

    @property
    def tools(self) -> List[Any]:
        """Get list of tools available to this agent.

        Returns:
            List of tool objects. Format depends on agent implementation.
            Default implementation returns empty list.
        """
        return []

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get agent metadata.

        Returns:
            Dictionary with agent metadata (model, config, etc.).
            Default implementation returns empty dict.
        """
        return {}
