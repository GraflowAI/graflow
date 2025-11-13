"""Google ADK agent wrapper for Graflow."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional

from .base import LLMAgent

if TYPE_CHECKING:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner as AdkRunner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types

try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner as AdkRunner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types as genai_types
    ADK_AVAILABLE = True
except ImportError:
    ADK_AVAILABLE = False

# OpenInference instrumentation for ADK tracing
try:
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor  # type: ignore[import-not-found]
    OPENINFERENCE_AVAILABLE = True
except ImportError:
    OPENINFERENCE_AVAILABLE = False
    GoogleADKInstrumentor = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)

# Global flag to track if instrumentation has been set up
_adk_instrumented = False


def setup_adk_tracing() -> None:
    """Setup OpenInference instrumentation for Google ADK.

    This enables automatic tracing of ADK agent calls to Langfuse via OpenTelemetry.
    Should be called once at application startup.

    Requires:
        - openinference-instrumentation-google-adk package
        - Langfuse tracing configured (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
        - LangFuseTracer active in workflow

    Example:
        ```python
        from graflow.llm.agents.adk_agent import setup_adk_tracing

        # Setup once at startup
        setup_adk_tracing()

        # Then use ADK agents in workflow
        from google.adk.agents import LlmAgent
        adk_agent = LlmAgent(name="assistant", model="gemini-2.0-flash-exp")
        agent = AdkLLMAgent(adk_agent, app_name=context.session_id)
        ```

    Note:
        - This function is idempotent (safe to call multiple times)
        - Instrumentation is global and affects all ADK agents
        - ADK traces will automatically nest under LangFuseTracer spans
    """
    global _adk_instrumented

    if _adk_instrumented:
        logger.debug("Google ADK instrumentation already set up")
        return

    if not OPENINFERENCE_AVAILABLE:
        logger.warning(
            "OpenInference instrumentation not available. "
            "Install with: pip install openinference-instrumentation-google-adk"
        )
        return

    try:
        if GoogleADKInstrumentor is not None:
            GoogleADKInstrumentor().instrument()  # type: ignore[misc]
            _adk_instrumented = True
            logger.info("Google ADK instrumentation enabled for tracing")
    except Exception as e:
        logger.warning(f"Failed to instrument Google ADK: {e}")

class AdkLLMAgent(LLMAgent):
    """Wrapper for Google ADK LlmAgent.

    This class wraps Google ADK's LlmAgent and provides Graflow integration.
    It uses delegation pattern to forward calls to the underlying LlmAgent.

    Google ADK agents support:
    - ReAct pattern (reasoning + tool calling)
    - Supervisor pattern (hierarchical sub-agents)
    - Streaming responses
    - Built-in Langfuse tracing integration

    Example:
        ```python
        from google.adk.agents import LlmAgent
        from graflow.llm.agents import AdkLLMAgent
        from graflow.core.context import ExecutionContext

        # Create context to get session_id
        context = ExecutionContext.create(...)

        # Create ADK agent with tools
        adk_agent = LlmAgent(
            name="supervisor",
            model="gemini-2.0-flash-exp",
            tools=[search_tool, calculator_tool],
            sub_agents=[analyst_agent, writer_agent]
        )

        # Wrap for Graflow with app_name from session_id
        agent = AdkLLMAgent(adk_agent, app_name=context.session_id)

        # Register in context
        context.register_llm_agent("supervisor", agent)

        # Use in task
        @task(inject_llm_agent="supervisor")
        def supervise_task(agent: LLMAgent, query: str) -> str:
            result = agent.run(query)
            return result["output"]
        ```

    Note:
        When enable_tracing=True in LLMConfig and LangFuseTracer is active,
        ADK will automatically detect OpenTelemetry context and create nested
        spans in Langfuse.
    """

    def __init__(
        self,
        adk_agent: LlmAgent,
        app_name: str,
        session_service: Optional[InMemorySessionService] = None,
        enable_tracing: bool = True,
    ):
        """Initialize AdkLLMAgent.

        Args:
            adk_agent: Google ADK LlmAgent instance
            app_name: Application name for Runner (typically workflow session_id).
                     This should be set to the workflow's session_id for proper identification.
            session_service: Session service (defaults to InMemorySessionService)
            enable_tracing: If True, automatically setup ADK tracing (default: True).
                          Set to False if you want to manually control instrumentation.

        Raises:
            RuntimeError: If Google ADK is not installed
            TypeError: If adk_agent is not a LlmAgent instance

        Example:
            ```python
            from graflow.core.context import ExecutionContext
            from google.adk.agents import LlmAgent

            # Create context to get session_id
            context = ExecutionContext.create(...)

            # Create agent with session_id as app_name
            adk_agent = LlmAgent(name="supervisor", model="gemini-2.0-flash-exp")
            agent = AdkLLMAgent(adk_agent, app_name=context.session_id)

            # Register agent
            context.register_llm_agent("supervisor", agent)
            ```

        Note:
            If enable_tracing=True (default), this will automatically call
            setup_adk_tracing() to enable OpenInference instrumentation.
        """
        if not ADK_AVAILABLE:
            raise RuntimeError(
                "Google ADK is not installed. "
                "Install with: pip install google-adk"
            )

        if not isinstance(adk_agent, LlmAgent):
            raise TypeError(
                f"adk_agent must be a LlmAgent instance, got {type(adk_agent)}"
            )

        # Setup tracing if enabled (idempotent - safe to call multiple times)
        if enable_tracing:
            setup_adk_tracing()

        self._adk_agent: LlmAgent = adk_agent
        self._app_name = app_name

        # Create session service if not provided
        if session_service is None:
            session_service = InMemorySessionService()

        # Store session service for future use
        self._session_service = session_service

        # Create Runner with app_name
        self._runner: AdkRunner = AdkRunner(  # type: ignore
            agent=adk_agent,
            app_name=self._app_name,
            session_service=session_service
        )

        # Default user for ADK execution
        self._default_user_id = "graflow-user"

    def run(
        self,
        input_text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Run the ADK agent synchronously.

        Args:
            input_text: Input query/prompt for the agent
            user_id: User ID for ADK session (defaults to "graflow-user")
            session_id: Session ID for ADK conversation history (defaults to generated UUID)
            **kwargs: Additional parameters forwarded to Runner.run()

        Returns:
            Dictionary with agent execution result:
            - "output": Final output text
            - "steps": List of execution events
            - "metadata": Additional metadata

        Example:
            ```python
            result = agent.run(
                "Analyze the sales data and create a report",
                session_id="conversation-123"
            )
            print(result["output"])
            ```

        Note:
            The session_id parameter here is for ADK's conversation history,
            separate from the app_name (set in __init__ from workflow trace_id).
        """
        try:
            import asyncio

            # Set defaults
            if user_id is None:
                user_id = self._default_user_id
            if session_id is None:
                # Generate a new session ID for this execution
                session_id = str(uuid.uuid4())

            # Create session in session service (async in ADK 1.0+)
            # Use asyncio.run() to execute async session creation synchronously
            asyncio.run(
                self._session_service.create_session(
                    app_name=self._app_name,
                    user_id=user_id,
                    session_id=session_id
                )
            )

            # Create Content from input text
            content = genai_types.Content(  # type: ignore
                role="user",
                parts=[genai_types.Part(text=input_text)]  # type: ignore
            )

            # Run agent via Runner (synchronous)
            # Runner.run() returns an iterable of events
            events = self._runner.run(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                **kwargs
            )

            # Collect all events and extract final response
            final_event = None
            all_events = []
            for event in events:
                all_events.append(event)
                if event.is_final_response():
                    final_event = event

            if final_event is None:
                raise RuntimeError("No final response from ADK agent")

            # Convert to standard format
            return self._convert_event_to_result(final_event, all_events)

        except Exception as e:
            logger.error(f"ADK agent execution failed: {e}")
            raise

    async def run_async(
        self,
        input_text: str,
        **kwargs: Any
    ) -> AsyncIterator[Any]:
        """Run the ADK agent asynchronously with streaming events.

        This method uses Runner.run_async() to provide true async execution
        with event streaming. Events include partial responses for streaming
        and final responses.

        Args:
            input_text: Input query/prompt for the agent
            **kwargs: Additional parameters forwarded to Runner.run_async()
                     Common parameters:
                     - user_id: User identifier (defaults to "graflow-user")
                     - session_id: Session identifier for conversation history
                     - invocation_id: Optional invocation identifier
                     - state_delta: Optional state changes
                     - run_config: Optional RunConfig

        Yields:
            ADK Runner events. Events have:
            - partial=True: Intermediate streaming chunks
            - is_final_response(): Final result marker
            - content: Event content

        Example:
            ```python
            async for event in agent.run_async("Tell me about Python"):
                if hasattr(event, 'partial') and event.partial:
                    # Streaming chunk
                    print(event.content, end="", flush=True)
                elif event.is_final_response():
                    # Final response
                    print(f"\\nFinal: {event.content}")
            ```
        """
        try:
            # Prepare parameters
            user_id = kwargs.pop("user_id", None)
            session_id = kwargs.pop("session_id", None)

            if user_id is None:
                user_id = self._default_user_id
            if session_id is None:
                session_id = str(uuid.uuid4())

            # Create Content for ADK
            content = genai_types.Content(  # type: ignore
                role="user",
                parts=[genai_types.Part(text=input_text)]  # type: ignore
            )

            # Run agent via Runner (asynchronous)
            # Runner.run_async() returns an async generator of events
            async for event in self._runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                **kwargs
            ):
                yield event

        except Exception as e:
            logger.error(f"ADK agent async execution failed: {e}")
            raise

    @property
    def name(self) -> str:
        """Get agent name from ADK agent."""
        return getattr(self._adk_agent, "name", "unknown")

    @property
    def tools(self) -> List[Any]:
        """Get list of tools from ADK agent."""
        return getattr(self._adk_agent, "tools", [])

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata from ADK agent."""
        return {
            "name": self.name,
            "model": getattr(self._adk_agent, "model", None),
            "tools_count": len(self.tools),
            "framework": "google-adk"
        }

    @classmethod
    def _from_adk_agent(cls, adk_agent: LlmAgent, app_name: str) -> AdkLLMAgent:
        """Create AdkLLMAgent from LlmAgent instance.

        This is used internally for deserialization from YAML.

        Args:
            adk_agent: Google ADK LlmAgent instance
            app_name: Application name for Runner (typically workflow session_id).

        Returns:
            AdkLLMAgent wrapper
        """
        return cls(adk_agent, app_name)

    def _convert_event_to_result(self, final_event: Any, all_events: List[Any]) -> Dict[str, Any]:
        """Convert ADK events to standard format.

        Args:
            final_event: The final response event from ADK Runner
            all_events: All events from the ADK Runner execution

        Returns:
            Standardized result dictionary with output, steps, and metadata
        """
        # Extract output text from final event
        output = ""
        if hasattr(final_event, "content") and final_event.content:
            if hasattr(final_event.content, "parts") and final_event.content.parts:
                # Get text from first part
                first_part = final_event.content.parts[0]
                if hasattr(first_part, "text"):
                    output = first_part.text or ""

        # Convert events to steps
        steps = []
        for event in all_events:
            step = {
                "type": type(event).__name__,
                "is_final": event.is_final_response() if hasattr(event, "is_final_response") else False,
                "is_partial": getattr(event, "partial", False),
            }
            # Add content if available
            if hasattr(event, "content") and event.content:
                if hasattr(event.content, "parts") and event.content.parts:
                    step["content"] = [
                        getattr(part, "text", str(part))
                        for part in event.content.parts
                    ]
            steps.append(step)

        return {
            "output": output,
            "steps": steps,
            "metadata": {
                "agent_name": self.name,
                "event_count": len(all_events),
            }
        }

    def _convert_adk_result(self, adk_result: Any) -> Dict[str, Any]:
        """Convert ADK result to standard format (legacy, for backwards compatibility).

        Args:
            adk_result: Result from ADK agent.run()

        Returns:
            Standardized result dictionary
        """
        # ADK result format may vary, handle common cases
        if isinstance(adk_result, dict):
            return {
                "output": adk_result.get("output", str(adk_result)),
                "steps": adk_result.get("steps", []),
                "metadata": adk_result.get("metadata", {})
            }
        elif isinstance(adk_result, str):
            return {
                "output": adk_result,
                "steps": [],
                "metadata": {}
            }
        else:
            return {
                "output": str(adk_result),
                "steps": [],
                "metadata": {}
            }
