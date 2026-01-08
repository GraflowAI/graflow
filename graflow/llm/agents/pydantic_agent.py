"""Pydantic AI agent wrapper for Graflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Generic, List, Optional, Type

from pydantic import BaseModel
from typing_extensions import TypeVar

from graflow.llm.agents.base import LLMAgent
from graflow.llm.agents.types import AgentResult, AgentStep

if TYPE_CHECKING:
    from pydantic_ai import Agent, AgentRunResult

logger = logging.getLogger(__name__)

try:
    from pydantic_ai import Agent

    PYDANTIC_AI_AVAILABLE = True
except ImportError as e:
    logger.warning("Pydantic AI is not installed. PydanticLLMAgent will not be available.", exc_info=e)
    PYDANTIC_AI_AVAILABLE = False

# Type variable for Agent output type (matches Pydantic AI's definition)
OutputDataT_co = TypeVar("OutputDataT_co", default=str, covariant=True)
"""Covariant type variable for the output data type of a run."""

# Global flag to track if instrumentation has been set up
_pydantic_ai_instrumented = False


def setup_pydantic_ai_tracing() -> None:
    """Setup Langfuse instrumentation for Pydantic AI.

    This enables automatic tracing of Pydantic AI agent calls to Langfuse via OpenTelemetry.
    Should be called once at application startup.

    Requires:
        - pydantic-ai package
        - Langfuse tracing configured (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY)
        - LangFuseTracer active in workflow (optional, for nesting under Graflow spans)

    Example:
        ```python
        from graflow.llm.agents.pydantic_agent import setup_pydantic_ai_tracing

        # Setup once at startup
        setup_pydantic_ai_tracing()

        # Then use Pydantic AI agents in workflow
        from pydantic_ai import Agent

        # Option 1: Agent with instrument=True (explicit)
        pydantic_agent = Agent(model='openai:gpt-4o', instrument=True)

        # Option 2: Agent without instrument flag (still traced via instrument_all())
        pydantic_agent = Agent(model='openai:gpt-4o')

        agent = PydanticLLMAgent(pydantic_agent, name="assistant")
        ```

    Note:
        - This function is idempotent (safe to call multiple times)
        - Calls `Agent.instrument_all()` which enables tracing GLOBALLY for all agents
        - Enables tracing even for user-provided agents created without `instrument=True`
        - Pydantic AI traces will automatically nest under LangFuseTracer spans via OpenTelemetry
        - Automatically traces agent calls, tool calls, and model API calls
        - This is particularly useful when users create their own Agent instances directly
    """
    global _pydantic_ai_instrumented  # noqa: PLW0603

    if _pydantic_ai_instrumented:
        logger.debug("Pydantic AI instrumentation already set up")
        return

    if not PYDANTIC_AI_AVAILABLE:
        logger.warning("Pydantic AI is not available. Install with: pip install pydantic-ai")
        return

    try:
        # Enable Langfuse instrumentation for all Pydantic AI agents
        # This must be called before creating any Agent instances with instrument=True
        Agent.instrument_all()

        _pydantic_ai_instrumented = True
        logger.info("Pydantic AI instrumentation enabled for tracing")

    except Exception as e:
        logger.warning(f"Failed to instrument Pydantic AI: {e}")


def create_pydantic_ai_agent_with_litellm(
    model: str,
    *,
    output_type: Optional[Type[BaseModel]] = None,
    instructions: Optional[str] = None,
    system_prompt: Optional[str] = None,
    name: Optional[str] = None,
    instrument: Optional[bool] = None,
    **kwargs: Any,
) -> Agent:
    """Create a Pydantic AI Agent with LiteLLM backend (optional convenience helper).

    This helper creates a Pydantic AI agent that routes requests through LiteLLM,
    enabling unified model access across providers. This is optional - users can
    create Agent instances directly and wrap them with PydanticLLMAgent.

    Args:
        model: Model identifier in LiteLLM format (e.g., 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet')
        output_type: Optional Pydantic model for structured output
        instructions: Optional instructions for the agent (recommended over system_prompt)
                     Instructions are not retained in message history across runs
        system_prompt: Optional system prompt (persists in message history)
        name: Optional name for the agent (useful for debugging/logging)
        instrument: Optional flag to enable Langfuse tracing (default: None, uses Pydantic AI default)
        **kwargs: Additional Agent parameters (model_settings, retries, tools, etc.)

    Returns:
        Pydantic AI Agent configured with LiteLLM backend

    Example:
        ```python
        from pydantic import BaseModel

        class AnalysisResult(BaseModel):
            sentiment: str
            confidence: float
            key_points: list[str]

        # Create agent with structured output and instructions
        agent = create_pydantic_ai_agent_with_litellm(
            'openai/gpt-4o',
            output_type=AnalysisResult,
            instructions="You are a data analyst. Analyze sentiment and extract key points.",
            name="sentiment-analyzer",
            instrument=True  # Enable tracing
        )

        # Wrap for Graflow
        wrapped = PydanticLLMAgent(agent, name="analyzer")
        ```

    Note:
        - API keys loaded from environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
        - Model names use LiteLLM format: 'provider/model' (e.g., 'openai/gpt-4o')
        - Prefer 'instructions' over 'system_prompt' for most use cases
        - This is just a convenience helper - you can create Agent instances any way you want
    """
    if not PYDANTIC_AI_AVAILABLE:
        raise RuntimeError("Pydantic AI is not installed. Install with: pip install pydantic-ai")

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.litellm import LiteLLMProvider

    # Create LiteLLM provider (API keys from environment)
    provider = LiteLLMProvider()

    # Create OpenAI-compatible model with LiteLLM backend
    llm_model = OpenAIChatModel(model, provider=provider, **kwargs)

    # Build agent kwargs - all parameters are optional
    agent_kwargs: Dict[str, Any] = {}

    # Add optional parameters only if provided (None values are omitted)
    if output_type is not None:
        agent_kwargs["output_type"] = output_type
    if instructions is not None:
        agent_kwargs["instructions"] = instructions
    if system_prompt is not None:
        agent_kwargs["system_prompt"] = system_prompt
    if name is not None:
        agent_kwargs["name"] = name
    if instrument is not None:
        agent_kwargs["instrument"] = instrument

    return Agent(llm_model, **agent_kwargs)


class PydanticLLMAgent(LLMAgent, Generic[OutputDataT_co]):
    """Wrapper for Pydantic AI Agent with type-safe output handling.

    This class wraps Pydantic AI's Agent and provides Graflow integration.
    It uses delegation pattern to forward calls to the underlying Agent.
    The class is generic in OutputDataT, preserving the type of Agent's output.

    Type Parameters:
        OutputDataT: The output type of the wrapped Agent (str, BaseModel, etc.)

    Pydantic AI agents support:
    - Multiple LLM providers (OpenAI, Anthropic, Google, etc.)
    - Type-safe structured outputs via Pydantic
    - Tool calling with decorator-based registration
    - Streaming responses
    - Built-in validation and error handling

    Example:
        ```python
        from pydantic_ai import Agent
        from graflow.llm.agents import PydanticLLMAgent
        from graflow.core.context import ExecutionContext
        from pydantic import BaseModel

        # Define structured output
        class AnalysisResult(BaseModel):
            sentiment: str
            confidence: float
            key_points: list[str]

        # Create Pydantic AI agent with structured output
        pydantic_agent = Agent(
            model='openai:gpt-4o',
            output_type=AnalysisResult,
            system_prompt="You are a data analyst.",
        )

        # Register a tool
        @pydantic_agent.tool
        def search_data(query: str) -> dict:
            return {"results": ["result1", "result2"]}

        # Wrap for Graflow - type is preserved as PydanticLLMAgent[AnalysisResult]
        agent = PydanticLLMAgent(pydantic_agent, name="analyzer")

        # Register in context
        context.register_llm_agent("analyzer", agent)

        # Use in task
        @task(inject_llm_agent="analyzer")
        def analyze_task(agent: LLMAgent, text: str) -> dict:
            result = agent.run(text)
            # result["output"] is AnalysisResult instance (type-safe!)
            return result["output"].model_dump()
        ```

    Note:
        - Output type is preserved: PydanticLLMAgent[T] wraps Agent[Any, T]
        - Structured output is controlled by Agent's output_type parameter
        - Tools are registered via @agent.tool decorator
        - Message history must be managed explicitly for multi-turn conversations
        - OpenTelemetry context is auto-detected for tracing integration
    """

    def __init__(
        self,
        agent: Agent[Any, OutputDataT_co],
        name: Optional[str] = None,
        enable_tracing: bool = True,
    ):
        """Initialize PydanticLLMAgent.

        Args:
            agent: Pydantic AI Agent instance (supports any output type)
            name: Agent name (defaults to "pydantic-agent")
            enable_tracing: If True, enable OpenTelemetry tracing (default: True)

        Raises:
            RuntimeError: If Pydantic AI is not installed
            TypeError: If agent is not a Pydantic AI Agent instance

        Example:
            ```python
            from pydantic_ai import Agent
            from pydantic import BaseModel

            class Output(BaseModel):
                answer: str
                confidence: float

            # Create Pydantic AI agent with structured output
            pydantic_agent = Agent(
                model='openai:gpt-4o-mini',
                output_type=Output,
                system_prompt="You are a helpful assistant."
            )

            # Wrap for Graflow - type is preserved
            agent = PydanticLLMAgent(pydantic_agent, name="assistant")
            # agent is now PydanticLLMAgent[Output]
            ```
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise RuntimeError("Pydantic AI is not installed. Install with: pip install pydantic-ai")

        if not isinstance(agent, Agent):
            raise TypeError(f"Expected pydantic_ai.Agent instance, got {type(agent)}")

        self._agent: Agent[Any, OutputDataT_co] = agent
        self._name = name or "pydantic-agent"
        self._enable_tracing = enable_tracing

        # Setup tracing if enabled (idempotent - safe to call multiple times)
        if enable_tracing:
            setup_pydantic_ai_tracing()

    def run(
        self,
        input_text: str,
        message_history: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Run the Pydantic AI agent synchronously.

        Args:
            input_text: Input query/prompt for the agent
            message_history: Optional message history from previous runs
                           (use result.new_messages() from previous AgentRunResult)
            **kwargs: Additional parameters forwarded to agent.run_sync()
                     Common parameters:
                     - model_settings: Override model settings
                     - usage_limits: Set token/request limits
                     - infer_name: Infer user name from input

        Returns:
            AgentResult:
                - output: Final output (str or Pydantic BaseModel based on output_type)
                - steps: Execution trace (messages exchanged)
                - metadata: Usage statistics and model info

        Example:
            ```python
            # First run
            result = agent.run("What is the capital of France?")
            print(result["output"])  # "Paris"

            # Follow-up with history
            result2 = agent.run(
                "What about Germany?",
                message_history=result["metadata"]["messages"]
            )
            ```
        """
        try:
            # Run agent synchronously
            run_result: AgentRunResult = self._agent.run_sync(input_text, message_history=message_history, **kwargs)

            # Convert to AgentResult
            return self._convert_run_result(run_result)

        except Exception as e:
            logger.error(f"Pydantic AI agent execution failed: {e}")
            raise

    async def run_async(
        self,
        input_text: str,
        message_history: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Run the Pydantic AI agent asynchronously with streaming.

        This method uses agent.run_stream() to provide async execution
        with event streaming. Events include text deltas and final output.

        Args:
            input_text: Input query/prompt for the agent
            message_history: Optional message history from previous runs
            **kwargs: Additional parameters forwarded to agent.run_stream()

        Yields:
            StreamedRunResult events with:
            - text deltas (via stream_text())
            - structured output (via stream_structured() if output_type is set)
            - final result

        Example:
            ```python
            async for chunk in agent.run_async("Tell me a story"):
                # Stream text as it arrives
                async for text in chunk.stream_text():
                    print(text, end="", flush=True)

                # Get final result
                final = await chunk.get_result()
                print(f"\\n\\nFinal: {final}")
            ```
        """
        try:
            # Run agent with streaming
            async with self._agent.run_stream(input_text, message_history=message_history, **kwargs) as stream:
                # Yield the stream for consumer to handle
                yield stream

        except Exception as e:
            logger.error(f"Pydantic AI agent async execution failed: {e}")
            raise

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name

    @property
    def tools(self) -> List[Any]:
        """Get list of tools registered with this agent.

        Returns:
            List of tool functions registered via @agent.tool decorator
        """
        # Pydantic AI stores tools internally
        # Access via agent._function_tools
        return getattr(self._agent, "_function_tools", [])

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get agent metadata.

        Returns:
            Dictionary with agent metadata (model, config, etc.)
        """
        return {
            "name": self.name,
            "model": getattr(self._agent, "model", None),
            "output_type": getattr(self._agent, "output_type", None),
            "tools_count": len(self.tools),
            "framework": "pydantic-ai",
        }

    def _format_message_part(self, part: Any) -> str:  # noqa: PLR0911
        """Format a message part for display.

        Args:
            part: Message part from Pydantic AI (TextPart, ToolCallPart, etc.)

        Returns:
            Formatted string representation of the part
        """
        # Use part_kind discriminator for type detection
        part_kind = getattr(part, "part_kind", None)

        if part_kind == "text":
            # TextPart: has 'content' attribute
            return str(getattr(part, "content", ""))
        elif part_kind == "thinking":
            # ThinkingPart: has 'content' attribute
            return f"[Thinking: {getattr(part, 'content', '')}]"
        elif part_kind == "tool-call":
            # ToolCallPart: has 'tool_name' attribute
            return f"[Tool: {getattr(part, 'tool_name', 'unknown')}]"
        elif part_kind == "system-prompt":
            # SystemPromptPart: has 'content' attribute
            return f"[System: {getattr(part, 'content', '')}]"
        elif part_kind == "user-prompt":
            # UserPromptPart: has 'content' attribute (can be str or list)
            user_content = getattr(part, "content", "")
            if isinstance(user_content, str):
                return user_content
            else:
                # Handle multimodal content
                return str(user_content)
        elif part_kind == "tool-return":
            # ToolReturnPart: tool execution result
            return f"[Tool Result: {getattr(part, 'tool_name', 'unknown')}]"
        elif part_kind == "file":
            # FilePart: binary content (images, documents, etc.)
            return "[File]"
        elif part_kind:
            # Other known part kinds
            return f"[{part_kind}]"
        else:
            # Unknown part type (fallback)
            return f"[Unknown: {type(part).__name__}]"

    def _convert_run_result(self, run_result: AgentRunResult) -> AgentResult:
        """Convert Pydantic AI RunResult to Graflow AgentResult.

        Args:
            run_result: RunResult from Pydantic AI agent execution

        Returns:
            AgentResult with output, steps, and metadata
        """
        # Extract output (can be str or Pydantic BaseModel)
        output: Any = run_result.output

        # Convert messages to steps
        steps: List[AgentStep] = []
        for msg in run_result.all_messages():
            # Determine message type from kind attribute
            msg_type = msg.kind

            step: AgentStep = {
                "type": msg_type,
                "is_final": False,
                "is_partial": False,
            }

            # Add content from message parts
            if msg.parts:
                content = [self._format_message_part(part) for part in msg.parts]
                step["content"] = content

            steps.append(step)

        # Mark last step as final
        if steps:
            steps[-1]["is_final"] = True

        # Build metadata
        metadata = {
            "agent_name": self.name,
            "framework": "pydantic-ai",
            "messages": run_result.new_messages(),  # For conversation history
        }

        # Add usage statistics if available
        usage = run_result.usage()
        if usage:
            metadata["usage"] = {
                "requests": usage.requests,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }

        return {
            "output": output,
            "steps": steps,
            "metadata": metadata,
        }
