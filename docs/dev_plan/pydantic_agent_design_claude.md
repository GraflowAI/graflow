# PydanticLLMAgent Design Document

## Overview

Design document for implementing `PydanticLLMAgent` - a Graflow LLM agent wrapper for [Pydantic AI](https://ai.pydantic.dev/).

### TL;DR - Key Design Decisions

1. **Simple Wrapper**: `PydanticLLMAgent` wraps any `pydantic_ai.Agent` instance - users create agents however they want
2. **Optional Helper**: `create_pydantic_ai_agent()` helper uses LiteLLM backend for unified routing (completely optional)
3. **Tracing**: Langfuse native support via `setup_pydantic_ai_tracing()` → calls `Agent.instrument_all()`
4. **Flexibility**: Three approaches to create agents:
   - **Direct**: `Agent('openai:gpt-4o', instrument=True)` - simplest
   - **Custom**: Full control with `OpenAIModel(api_key=..., base_url=...)`
   - **Helper**: `create_pydantic_ai_agent('openai/gpt-4o')` - LiteLLM backend

All approaches produce valid `Agent` instances that wrap with `PydanticLLMAgent(agent, name="...")`.

### Purpose

- Provide an alternative to `AdkLLMAgent` using Pydantic AI framework
- Leverage Pydantic AI's type-safe agent system with structured outputs
- Support multiple LLM providers (OpenAI, Anthropic, Google, etc.) through unified interface
- Maintain compatibility with Graflow's `LLMAgent` base class
- Enable seamless integration with existing LLM infrastructure (tracing, serialization)

### Why Pydantic AI?

**Advantages over Google ADK:**
- **Type Safety**: Full mypy/pyright support with generic typing `Agent[DepsType, OutputType]`
- **Multi-Provider**: Native support for OpenAI, Anthropic, Google, Bedrock, Groq, Mistral, etc.
- **Structured Output**: Built-in Pydantic validation for agent responses
- **Lightweight**: Minimal dependencies compared to ADK
- **Python-First**: Pythonic API design with decorators and async/await patterns
- **Tool Flexibility**: Simple decorator-based tool registration
- **Streaming Support**: First-class streaming for text and structured outputs

**Use Cases:**
- Workflows requiring multi-provider support (fallback strategies, cost optimization)
- Type-safe structured outputs for downstream processing
- Simpler agent setup without ADK's App/Runner/Session complexity
- Rapid prototyping with decorator-based tools

---

## Architecture

### Module Structure

```
graflow/llm/agents/
├── __init__.py
├── base.py              # LLMAgent base class (existing)
├── adk_agent.py         # Google ADK wrapper (existing)
└── pydantic_agent.py    # NEW: Pydantic AI wrapper
```

### Class Hierarchy

```
LLMAgent (ABC)
├── AdkLLMAgent       # Google ADK implementation
└── PydanticLLMAgent  # Pydantic AI implementation (NEW)
```

### Integration with Graflow

```
┌────────────────────────────────────────────────────────────────┐
│ Graflow Execution Layer                                       │
│                                                                │
│  ExecutionContext                                             │
│    ├─ llm_client (LLMClient instance)                         │
│    └─ llm_agents: Dict[str, LLMAgent]                         │
│         ├─ "supervisor" → AdkLLMAgent                         │
│         └─ "analyzer" → PydanticLLMAgent  ← NEW               │
│                                                                │
│  Task: @task(inject_llm_agent="analyzer")                     │
└────────────────────────────────────────────────────────────────┘
                        │
                        │ Agent execution
                        ▼
┌────────────────────────────────────────────────────────────────┐
│ PydanticLLMAgent (Wrapper)                                    │
│                                                                │
│  ├─ _agent: Agent[None, OutputT]  ← Pydantic AI Agent        │
│  ├─ run(input_text) → AgentResult                            │
│  │   └─ agent.run_sync(input_text) → RunResult              │
│  │       └─ Convert to AgentResult                           │
│  │                                                            │
│  └─ run_async(input_text) → AsyncIterator[Any]               │
│      └─ agent.run_stream(input_text)                         │
│          └─ Yield StreamedRunResult events                    │
└────────────────────────────────────────────────────────────────┘
                        │
                        │ LLM calls via Pydantic AI models
                        ▼
┌────────────────────────────────────────────────────────────────┐
│ Pydantic AI Model Layer                                       │
│                                                                │
│  Model providers (via pydantic_ai.models)                     │
│    ├─ OpenAI (openai:gpt-4o, openai:gpt-4o-mini)             │
│    ├─ Anthropic (anthropic:claude-3-5-sonnet-20241022)       │
│    ├─ Google (gemini-1.5-pro, gemini-2.0-flash-exp)          │
│    └─ Others (Groq, Mistral, Bedrock, etc.)                  │
└────────────────────────────────────────────────────────────────┘
                        │
                        │ OpenTelemetry context propagation
                        ▼
┌────────────────────────────────────────────────────────────────┐
│ Langfuse (via OpenTelemetry)                                  │
│                                                                │
│  Unified tracing with same trace_id                           │
│  - Graflow tasks: LangFuseTracer                             │
│  - LLM calls: Auto-detected from OpenTelemetry context       │
└────────────────────────────────────────────────────────────────┘
```

---

## Core Design

### 0. Helper Function for Agent Creation (Optional)

**Decision**: Provide `create_pydantic_ai_agent()` helper that uses LiteLLM backend by default

**Rationale**:
- Convenient helper for creating Pydantic AI agents with LiteLLM routing
- Uses `LiteLLMProvider` for unified model access (consistent with Graflow's `LLMClient`)
- API keys automatically loaded from environment variables
- **Completely optional**: Users can create `Agent` instances any way they want

**Implementation**:
```python
def create_pydantic_ai_agent(
    model: str,
    *,
    output_type: Optional[Type[BaseModel]] = None,
    system_prompt: Optional[str] = None,
    instrument: bool = True,
    **kwargs: Any
) -> Agent:
    """Create a Pydantic AI Agent with LiteLLM backend (optional convenience helper).

    This helper creates a Pydantic AI agent that routes requests through LiteLLM,
    enabling unified model access across providers. This is optional - users can
    create Agent instances directly and wrap them with PydanticLLMAgent.

    Args:
        model: Model identifier in LiteLLM format (e.g., 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet')
        output_type: Optional Pydantic model for structured output
        system_prompt: System instructions for the agent
        instrument: Enable Langfuse tracing (default: True)
        **kwargs: Additional OpenAIChatModel parameters

    Returns:
        Pydantic AI Agent configured with LiteLLM backend

    Example:
        ```python
        # Uses OPENAI_API_KEY from environment
        agent = create_pydantic_ai_agent(
            'openai/gpt-4o',
            output_type=AnalysisResult,
            system_prompt="You are a data analyst."
        )

        # Wrap for Graflow
        wrapped = PydanticLLMAgent(agent, name="analyzer")
        ```

    Note:
        - API keys loaded from environment (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
        - Model names use LiteLLM format: 'provider/model' (e.g., 'openai/gpt-4o')
        - This is just a convenience helper - you can create Agent instances any way you want
    """
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.litellm import LiteLLMProvider

    # Create LiteLLM provider (API keys from environment)
    provider = LiteLLMProvider()

    # Create OpenAI-compatible model with LiteLLM backend
    llm_model = OpenAIChatModel(
        model,
        provider=provider,
        **kwargs
    )

    # Create agent
    return Agent(
        llm_model,
        output_type=output_type,
        system_prompt=system_prompt,
        instrument=instrument
    )
```

**Alternative Approaches (All Valid)**:

**1. Direct Agent Creation (no helper)**:
```python
from pydantic_ai import Agent

# Direct provider - simplest approach
agent = Agent(
    'openai:gpt-4o',  # Direct syntax: 'provider:model'
    output_type=OutputSchema,
    system_prompt="Instructions here",
    instrument=True
)
wrapped = PydanticLLMAgent(agent, name="my_agent")
```

**2. Custom Model Configuration**:
```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

# Full control over model configuration
model = OpenAIModel(
    model='gpt-4o',
    api_key=custom_key,
    base_url=custom_endpoint,
    temperature=0.7
)
agent = Agent(model, output_type=OutputSchema, instrument=True)
wrapped = PydanticLLMAgent(agent, name="custom_agent")
```

**3. Using Helper (LiteLLM backend)**:
```python
from graflow.llm.agents import create_pydantic_ai_agent, PydanticLLMAgent

# Convenient helper with LiteLLM routing
agent = create_pydantic_ai_agent(
    'openai/gpt-4o',  # LiteLLM format: 'provider/model'
    output_type=OutputSchema
)
wrapped = PydanticLLMAgent(agent, name="my_agent")
```

**When to Use Each Approach**:
- ✅ **Direct Agent**: Simple single-provider use cases
- ✅ **Custom Model**: Need fine-grained control over configuration
- ✅ **Helper**: Want LiteLLM routing for fallbacks, unified access, or cost optimization

---

### 1. Class Definition

```python
"""Pydantic AI agent wrapper for Graflow."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from graflow.llm.agents.base import LLMAgent
from graflow.llm.agents.types import AgentResult, AgentStep

if TYPE_CHECKING:
    from pydantic_ai import Agent, RunResult

logger = logging.getLogger(__name__)

try:
    from pydantic_ai import Agent
    PYDANTIC_AI_AVAILABLE = True
except ImportError as e:
    logger.warning("Pydantic AI is not installed. PydanticLLMAgent will not be available.", exc_info=e)
    PYDANTIC_AI_AVAILABLE = False

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
    global _pydantic_ai_instrumented

    if _pydantic_ai_instrumented:
        logger.debug("Pydantic AI instrumentation already set up")
        return

    if not PYDANTIC_AI_AVAILABLE:
        logger.warning(
            "Pydantic AI is not available. "
            "Install with: pip install pydantic-ai"
        )
        return

    try:
        # Enable Langfuse instrumentation for all Pydantic AI agents
        # This must be called before creating any Agent instances with instrument=True
        Agent.instrument_all()

        _pydantic_ai_instrumented = True
        logger.info("Pydantic AI instrumentation enabled for tracing")

    except Exception as e:
        logger.warning(f"Failed to instrument Pydantic AI: {e}")


class PydanticLLMAgent(LLMAgent):
    """Wrapper for Pydantic AI Agent.

    This class wraps Pydantic AI's Agent and provides Graflow integration.
    It uses delegation pattern to forward calls to the underlying Agent.

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

        # Wrap for Graflow
        agent = PydanticLLMAgent(pydantic_agent, name="analyzer")

        # Register in context
        context.register_llm_agent("analyzer", agent)

        # Use in task
        @task(inject_llm_agent="analyzer")
        def analyze_task(agent: LLMAgent, text: str) -> dict:
            result = agent.run(text)
            # result["output"] is AnalysisResult instance
            return result["output"].model_dump()
        ```

    Note:
        - Structured output is controlled by Agent's output_type parameter
        - Tools are registered via @agent.tool decorator
        - Message history must be managed explicitly for multi-turn conversations
        - OpenTelemetry context is auto-detected for tracing integration
    """

    def __init__(
        self,
        agent: Agent,
        name: Optional[str] = None,
        enable_tracing: bool = True,
    ):
        """Initialize PydanticLLMAgent.

        Args:
            agent: Pydantic AI Agent instance
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

            # Create Pydantic AI agent
            pydantic_agent = Agent(
                model='openai:gpt-4o-mini',
                output_type=Output,
                system_prompt="You are a helpful assistant."
            )

            # Wrap for Graflow
            agent = PydanticLLMAgent(pydantic_agent, name="assistant")
            ```
        """
        if not PYDANTIC_AI_AVAILABLE:
            raise RuntimeError(
                "Pydantic AI is not installed. "
                "Install with: pip install pydantic-ai"
            )

        if not isinstance(agent, Agent):
            raise TypeError(
                f"Expected pydantic_ai.Agent instance, got {type(agent)}"
            )

        self._agent: Agent = agent
        self._name = name or "pydantic-agent"
        self._enable_tracing = enable_tracing

        # Setup tracing if enabled (idempotent - safe to call multiple times)
        if enable_tracing:
            setup_pydantic_ai_tracing()

    def run(
        self,
        input_text: str,
        message_history: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> AgentResult:
        """Run the Pydantic AI agent synchronously.

        Args:
            input_text: Input query/prompt for the agent
            message_history: Optional message history from previous runs
                           (use result.new_messages() from previous RunResult)
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
            run_result: RunResult = self._agent.run_sync(
                input_text,
                message_history=message_history,
                **kwargs
            )

            # Convert to AgentResult
            return self._convert_run_result(run_result)

        except Exception as e:
            logger.error(f"Pydantic AI agent execution failed: {e}")
            raise

    async def run_async(
        self,
        input_text: str,
        message_history: Optional[List[Any]] = None,
        **kwargs: Any
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
            - structured output (via stream_output() if output_type is set)
            - final result

        Example:
            ```python
            async for chunk in agent.run_async("Tell me a story"):
                # Stream text as it arrives
                async for text in chunk.stream_text():
                    print(text, end="", flush=True)

                # Get final result
                final = await chunk.get_result()
                print(f"\n\nFinal: {final}")
            ```
        """
        try:
            # Run agent with streaming
            async with self._agent.run_stream(
                input_text,
                message_history=message_history,
                **kwargs
            ) as stream:
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
        return getattr(self._agent, '_function_tools', [])

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get agent metadata.

        Returns:
            Dictionary with agent metadata (model, config, etc.)
        """
        return {
            "name": self.name,
            "model": getattr(self._agent, 'model', None),
            "output_type": self._agent.output_type,  # Use public property
            "tools_count": len(self.tools),
            "framework": "pydantic-ai"
        }

    def _convert_run_result(self, run_result: RunResult) -> AgentResult:
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
            step: AgentStep = {
                "type": msg.role,
                "is_final": False,
                "is_partial": False,
            }

            # Add content from message parts
            if hasattr(msg, 'parts'):
                content = []
                for part in msg.parts:
                    if hasattr(part, 'content'):
                        content.append(str(part.content))
                    elif hasattr(part, 'tool_name'):
                        # Tool call
                        content.append(f"[Tool: {part.tool_name}]")
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
        if hasattr(run_result, 'usage'):
            metadata["usage"] = {
                "requests": run_result.usage.requests,
                "request_tokens": run_result.usage.request_tokens,
                "response_tokens": run_result.usage.response_tokens,
                "total_tokens": run_result.usage.total_tokens,
            }

        return {
            "output": output,
            "steps": steps,
            "metadata": metadata,
        }
```

---

## Key Design Decisions

### 1. Agent Wrapping Pattern

**Decision**: Wrap `pydantic_ai.Agent` directly (no additional abstraction layer)

**Rationale**:
- Pydantic AI's `Agent` is already a high-level interface
- Tools are registered via decorators on the Agent instance
- No need for App/Runner/Session complexity like ADK
- Direct delegation maintains simplicity

**Alternative Considered**: Create separate tool registry
- **Rejected**: Pydantic AI's decorator pattern is clean and Pythonic

### 2. Structured Output Handling

**Decision**: Use Pydantic AI's `output_type` parameter for structured outputs

**Rationale**:
- Native Pydantic validation built into framework
- Type-safe with generic `Agent[DepsType, OutputT]`
- Automatic validation and error handling
- No need for manual JSON parsing like ADK

**Example**:
```python
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    key_points: list[str]

agent = Agent(
    model='openai:gpt-4o',
    output_type=AnalysisResult,  # Type-safe structured output
)
```

### 3. Tool Registration

**Decision**: Use Pydantic AI's decorator-based tool registration

**Rationale**:
- Pythonic and intuitive
- Type annotations provide automatic validation
- Tools bound to Agent instance (no global registry)

**Example**:
```python
@agent.tool
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the database for relevant records."""
    return db.search(query, limit)
```

### 4. Message History Management

**Decision**: Explicit message history passing (not session-based like ADK)

**Rationale**:
- Pydantic AI uses explicit `message_history` parameter
- Simpler than ADK's session service
- More control over conversation context
- Stateless by default

**Example**:
```python
result1 = agent.run("First question")
result2 = agent.run(
    "Follow-up",
    message_history=result1["metadata"]["messages"]
)
```

### 5. Streaming Support

**Decision**: Implement `run_async()` with Pydantic AI's `run_stream()`

**Rationale**:
- Pydantic AI has first-class streaming support
- `StreamedRunResult` provides both text and structured output streams
- Async context manager pattern is clean

**Example**:
```python
async with agent.run_stream("Tell me a story") as stream:
    async for text in stream.stream_text():
        print(text, end="", flush=True)

    result = await stream.get_result()
```

### 6. Tracing Integration

**Decision**: Use Langfuse's native Pydantic AI instrumentation via `Agent.instrument_all()`

**Rationale**:
- Langfuse officially supports Pydantic AI (https://langfuse.com/integrations/frameworks/pydantic-ai)
- Uses OpenTelemetry under the hood for context propagation
- Automatic tracing of agent calls, tool calls, and model calls
- Consistent with ADK's tracing pattern (setup once at startup)
- No manual trace_id passing needed

**Implementation**:
```python
def setup_pydantic_ai_tracing() -> None:
    """Setup Langfuse instrumentation for Pydantic AI.

    Should be called once at application startup.
    """
    global _pydantic_ai_instrumented

    if _pydantic_ai_instrumented:
        return

    try:
        from pydantic_ai.agent import Agent
        Agent.instrument_all()
        _pydantic_ai_instrumented = True
        logger.info("Pydantic AI instrumentation enabled for tracing")
    except Exception as e:
        logger.warning(f"Failed to instrument Pydantic AI: {e}")
```

**Note**: Each agent must be created with `instrument=True` parameter to participate in tracing.

---

## Comparison: PydanticLLMAgent vs AdkLLMAgent

| Feature | AdkLLMAgent | PydanticLLMAgent |
|---------|-------------|------------------|
| **Framework** | Google ADK | Pydantic AI |
| **Initialization** | Complex (App, Runner, Session) | Simple (Agent instance) |
| **Multi-Provider** | Via LiteLLM integration | Native (OpenAI, Anthropic, Google, etc.) |
| **Structured Output** | `output_schema` + JSON parsing | `output_type` with native Pydantic validation |
| **Tool Registration** | Function-based | Decorator-based (`@agent.tool`) |
| **Message History** | Session-based (InMemorySessionService) | Explicit `message_history` parameter |
| **Streaming** | `run_async()` with events | `run_stream()` with async context manager |
| **Type Safety** | Limited | Full mypy/pyright support with generics |
| **Tracing** | OpenInference + manual patching | Langfuse native via `Agent.instrument_all()` |
| **Dependencies** | Heavy (google-adk, nest-asyncio) | Light (pydantic-ai, httpx) |
| **Setup Complexity** | High (instrumentation, patches) | Low (direct Agent instantiation) |
| **Use Case** | Google ecosystem, complex workflows | Multi-provider, type-safe, rapid development |

**When to Use Each**:

**Use AdkLLMAgent when**:
- ✅ Using Google Vertex AI or Gemini models exclusively
- ✅ Need complex sub-agent hierarchies (supervisor pattern)
- ✅ Already invested in Google ADK ecosystem
- ✅ Need ADK-specific features (e.g., advanced state management)

**Use PydanticLLMAgent when**:
- ✅ Need multi-provider support (OpenAI, Anthropic, Google, etc.)
- ✅ Want strong type safety with Pydantic validation
- ✅ Prefer simpler setup without App/Runner/Session complexity
- ✅ Need rapid prototyping and iteration
- ✅ Want decorator-based tool registration
- ✅ Need structured outputs with native Pydantic validation
- ✅ Using LiteLLM for unified routing and fallbacks

**Hybrid Approach** (both in same workflow):
```python
# Use ADK for Google-specific agents
adk_supervisor = AdkLLMAgent(google_adk_agent, app_name=session_id)

# Use Pydantic AI for multi-provider agents
claude_analyzer = PydanticLLMAgent(
    Agent('anthropic:claude-3-5-sonnet-20241022', instrument=True),
    name="analyzer"
)

# Register both in same context
context.register_llm_agent("supervisor", adk_supervisor)
context.register_llm_agent("analyzer", claude_analyzer)
```

---

## Implementation Roadmap

### Phase 1: Core Implementation ✅ (Design Complete)

- [ ] Implement `PydanticLLMAgent` class in `graflow/llm/agents/pydantic_agent.py`
- [ ] Add `PYDANTIC_AI_AVAILABLE` check with graceful degradation
- [ ] Implement `setup_pydantic_ai_tracing()` function (calls `Agent.instrument_all()`)
- [ ] Implement `run()` method with `agent.run_sync()`
- [ ] Implement `_convert_run_result()` for AgentResult conversion
- [ ] Add `name`, `tools`, `metadata` properties

### Phase 2: Helper & Async Streaming

- [ ] Implement optional `create_pydantic_ai_agent()` helper (uses LiteLLM backend)
- [ ] Implement `run_async()` with `agent.run_stream()`
- [ ] Handle `StreamedRunResult` yielding
- [ ] Add examples for both direct and helper approaches

### Phase 3: Tracing Integration ✅ (Resolved)

- [x] ~~Investigate Pydantic AI's OpenTelemetry support~~ (Langfuse officially supports Pydantic AI)
- [x] ~~Implement tracing propagation~~ (Automatic via `Agent.instrument_all()`)
- [ ] Add Langfuse integration tests
- [ ] Document tracing behavior and verify OpenTelemetry context propagation

### Phase 4: Advanced Features

- [ ] Message history management utilities
- [ ] Usage limits integration
- [ ] Model settings override support
- [ ] Error handling improvements (UnexpectedModelBehavior)

### Phase 5: Testing & Documentation

- [ ] Unit tests for PydanticLLMAgent
- [ ] Integration tests with different providers
- [ ] Comparison tests vs AdkLLMAgent
- [ ] Usage examples in `examples/llm/`
- [ ] Update `llm_integration_design.md` with PydanticLLMAgent section

---

## Usage Examples

### Example 1: Simple Agent with Structured Output (Direct)

```python
from pydantic_ai import Agent
from pydantic import BaseModel
from graflow.llm.agents import PydanticLLMAgent
from graflow.core.decorators import task

# Define structured output
class SentimentAnalysis(BaseModel):
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float  # 0.0 to 1.0
    explanation: str

# Create Pydantic AI agent (direct provider syntax)
pydantic_agent = Agent(
    model='openai:gpt-4o-mini',  # Direct: 'provider:model'
    output_type=SentimentAnalysis,
    system_prompt="You are a sentiment analysis expert.",
    instrument=True  # Enable tracing
)

# Wrap for Graflow
agent = PydanticLLMAgent(pydantic_agent, name="sentiment_analyzer")

# Register in context
context.register_llm_agent("sentiment_analyzer", agent)

# Use in task
@task(inject_llm_agent="sentiment_analyzer")
def analyze_sentiment(agent: LLMAgent, text: str) -> dict:
    result = agent.run(text)
    # result["output"] is SentimentAnalysis instance (validated!)
    output: SentimentAnalysis = result["output"]
    return {
        "sentiment": output.sentiment,
        "confidence": output.confidence,
        "explanation": output.explanation
    }
```

### Example 1b: Same Agent with Helper (LiteLLM Backend)

```python
from pydantic import BaseModel
from graflow.llm.agents import PydanticLLMAgent, create_pydantic_ai_agent
from graflow.core.decorators import task

# Define structured output
class SentimentAnalysis(BaseModel):
    sentiment: str
    confidence: float
    explanation: str

# Create agent using helper (uses LiteLLM backend for unified routing)
pydantic_agent = create_pydantic_ai_agent(
    model='openai/gpt-4o-mini',  # LiteLLM format: 'provider/model'
    output_type=SentimentAnalysis,
    system_prompt="You are a sentiment analysis expert."
)

# Wrap for Graflow (same as direct approach)
agent = PydanticLLMAgent(pydantic_agent, name="sentiment_analyzer")

# Use in task (same as before)
@task(inject_llm_agent="sentiment_analyzer")
def analyze_sentiment(agent: LLMAgent, text: str) -> dict:
    result = agent.run(text)
    output: SentimentAnalysis = result["output"]
    return output.model_dump()
```

**Note**: Both Example 1 (direct) and Example 1b (helper) produce the same result. The helper just provides convenient LiteLLM routing.

### Example 2: Agent with Tools

```python
from pydantic_ai import Agent, RunContext
from graflow.llm.agents import PydanticLLMAgent

# Create agent
pydantic_agent = Agent(
    model='anthropic:claude-3-5-sonnet-20241022',
    system_prompt="You are a data analyst with access to database tools."
)

# Register tools via decorator
@pydantic_agent.tool
def search_customers(ctx: RunContext, query: str, limit: int = 10) -> list[dict]:
    """Search customer database."""
    # Tool implementation
    return db.search("customers", query, limit)

@pydantic_agent.tool
def get_sales_data(ctx: RunContext, customer_id: int, days: int = 30) -> dict:
    """Get sales data for a customer."""
    return db.get_sales(customer_id, days)

# Wrap and use
agent = PydanticLLMAgent(pydantic_agent, name="data_analyst")
result = agent.run("Find top 5 customers by sales in last 30 days")
```

### Example 3: Multi-Provider Fallback Strategy (Using Helper)

```python
from graflow.llm.agents import PydanticLLMAgent, create_pydantic_ai_agent
from graflow.core.workflow import workflow
from graflow.core.context import ExecutionContext

# High-performance agent (primary)
# Using helper with LiteLLM backend for unified routing
gpt4_agent = create_pydantic_ai_agent(
    model='openai/gpt-4o',
    system_prompt="You are an expert analyst."
)
primary = PydanticLLMAgent(gpt4_agent, name="primary")

# Cost-effective agent (fallback)
claude_agent = create_pydantic_ai_agent(
    model='anthropic/claude-3-5-sonnet-20241022',
    system_prompt="You are an expert analyst."
)
fallback = PydanticLLMAgent(claude_agent, name="fallback")

@task
def analyze_with_fallback(text: str, ctx: ExecutionContext) -> str:
    try:
        agent = ctx.get_llm_agent("primary")
        result = agent.run(text)
        return result["output"]
    except Exception as e:
        logger.warning(f"Primary agent failed: {e}, using fallback")
        agent = ctx.get_llm_agent("fallback")
        result = agent.run(text)
        return result["output"]
```

### Example 4: Conversation with History

```python
@task(inject_llm_agent="assistant")
def multi_turn_conversation(agent: LLMAgent) -> dict:
    # First turn
    result1 = agent.run("What is the capital of France?")
    print(result1["output"])  # "The capital of France is Paris."

    # Second turn with history
    history = result1["metadata"]["messages"]
    result2 = agent.run(
        "What about its population?",
        message_history=history
    )
    print(result2["output"])  # "Paris has a population of approximately 2.2 million..."

    return {
        "first_response": result1["output"],
        "second_response": result2["output"]
    }
```

### Example 5: Streaming Responses

```python
@task(inject_llm_agent="writer")
async def stream_story(agent: LLMAgent, prompt: str):
    """Stream a story in real-time."""
    async for stream in agent.run_async(prompt):
        # Stream text chunks
        async for text in stream.stream_text():
            print(text, end="", flush=True)

        # Get final result
        final_result = await stream.get_result()
        print(f"\n\n--- Complete ---")
        print(f"Total tokens: {final_result.usage.total_tokens}")
```

---

## Open Questions for Discussion

### ~~1. OpenTelemetry Integration~~ ✅ RESOLVED

**Resolution**: Langfuse officially supports Pydantic AI via `Agent.instrument_all()`
- Call once at startup: `setup_pydantic_ai_tracing()`
- Create agents with `instrument=True`
- Automatic OpenTelemetry context propagation
- Traces nest under LangFuseTracer spans

### 2. Dependencies Type Parameter

**Question**: Should we support Pydantic AI's `deps_type` (dependency injection)?

**Context**: Pydantic AI agents can inject dependencies via `Agent[DepsType, OutputT]`

**Options**:
- **A**: Ignore for now (use `Agent[None, OutputT]`)
- **B**: Support via `PydanticLLMAgent` constructor parameter
- **C**: Map to Graflow's ExecutionContext

**Recommendation**: Start with Option A, add Option B if needed

### 3. Model Provider Configuration

**Question**: How should users configure LLM providers (API keys, endpoints)?

**Options**:
- **A**: Environment variables only (current pattern)
- **B**: Pass via `PydanticLLMAgent` constructor
- **C**: Centralized LLM config in ExecutionContext

**Recommendation**: Option A for consistency, document required env vars

### 4. Agent Serialization

**Question**: How to serialize/deserialize Pydantic AI agents for distributed execution?

**Context**: ADK agents serialize as YAML, Pydantic AI agents are Python objects

**Options**:
- **A**: Cloudpickle (same as Graflow tasks)
- **B**: Custom serialization (recreate agent from config)
- **C**: Raise error if distributed execution attempted

**Recommendation**: Start with Option A (cloudpickle), evaluate stability

### 5. Tool State Management

**Question**: How to handle stateful tools in distributed workers?

**Context**: Tools registered via decorators may capture local state

**Options**:
- **A**: Require stateless tools only
- **B**: Serialize tool state via ExecutionContext
- **C**: Document limitation, provide guidelines

**Recommendation**: Option A + C (document best practices)

### 6. Error Handling Strategy

**Question**: How to handle Pydantic AI's `UnexpectedModelBehavior` errors?

**Options**:
- **A**: Wrap in generic RuntimeError
- **B**: Create custom GraflowAgentError hierarchy
- **C**: Pass through original exception

**Recommendation**: Option B for consistency with Graflow error handling

### 7. Usage Limits Integration

**Question**: Should we expose Pydantic AI's `UsageLimits` (token limits)?

**Options**:
- **A**: Not initially (users can set on Agent)
- **B**: Add to `run()` method signature
- **C**: Configure at PydanticLLMAgent level

**Recommendation**: Option A, add Option B if common use case

---

## Testing Strategy

### Unit Tests (`tests/llm/test_pydantic_agent.py`)

```python
import pytest
from pydantic_ai import Agent
from pydantic import BaseModel
from graflow.llm.agents.pydantic_agent import PydanticLLMAgent

class OutputSchema(BaseModel):
    answer: str
    confidence: float

@pytest.fixture
def simple_agent():
    agent = Agent(
        model='test',  # Mock model
        output_type=OutputSchema,
        system_prompt="Test agent"
    )
    return PydanticLLMAgent(agent, name="test")

def test_initialization(simple_agent):
    assert simple_agent.name == "test"
    assert simple_agent.metadata["framework"] == "pydantic-ai"

def test_run_with_structured_output(simple_agent, monkeypatch):
    # Mock pydantic_ai.Agent.run_sync
    mock_result = MockRunResult(
        output=OutputSchema(answer="42", confidence=0.95)
    )
    monkeypatch.setattr(simple_agent._agent, 'run_sync', lambda *args, **kwargs: mock_result)

    result = simple_agent.run("What is the answer?")
    assert isinstance(result["output"], OutputSchema)
    assert result["output"].answer == "42"

def test_tool_registration():
    agent = Agent(model='test')

    @agent.tool
    def search(query: str) -> str:
        return f"Results for {query}"

    wrapped = PydanticLLMAgent(agent, name="test")
    assert len(wrapped.tools) == 1
```

### Integration Tests (`tests/llm/test_pydantic_agent_integration.py`)

```python
import pytest
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from pydantic_ai import Agent
from graflow.llm.agents import PydanticLLMAgent

@pytest.mark.integration
@pytest.mark.skipif(not PYDANTIC_AI_AVAILABLE, reason="pydantic-ai not installed")
def test_agent_in_workflow(openai_api_key):
    # Create agent
    pydantic_agent = Agent(
        model='openai:gpt-4o-mini',
        system_prompt="You are a test assistant."
    )
    agent = PydanticLLMAgent(pydantic_agent, name="test_agent")

    # Create context and register
    context = ExecutionContext.create(...)
    context.register_llm_agent("test_agent", agent)

    # Use in task
    @task(inject_llm_agent="test_agent")
    def test_task(agent: LLMAgent) -> str:
        result = agent.run("Say 'Hello, Graflow!'")
        return result["output"]

    result = test_task.run()
    assert "hello" in result.lower()
```

### Provider Tests

```python
@pytest.mark.parametrize("model", [
    "openai:gpt-4o-mini",
    "anthropic:claude-3-5-sonnet-20241022",
    "gemini-1.5-flash",
])
def test_multi_provider_support(model, api_keys):
    agent = Agent(model=model)
    wrapped = PydanticLLMAgent(agent, name=f"test_{model}")

    result = wrapped.run("Say 'test'")
    assert result["output"]
```

---

## Dependencies & Installation

### Required Dependencies

**Current `pyproject.toml` configuration:**

```toml
[project]
dependencies = [
    # ... other deps ...
    "litellm>=1.79.1",   # Core dependency (already available)
    "pydantic>=2.10.0",  # Core dependency (already available)
]

[project.optional-dependencies]
pydantic-ai = [
    "pydantic-ai-slim>=1.33.0",  # Slim version without heavy dependencies
]
adk = [
    "google-adk>=1.17.0",
    # ... adk deps ...
]
all = [
    "graflow[standard]",  # Includes tracing, api, visualization, etc.
    "graflow[pydantic-ai]"
]
```

**Notes:**
- `pydantic-ai-slim` is used (not full `pydantic-ai`) to minimize dependencies
- `litellm` and `pydantic` are core dependencies, so they're always available
- `create_pydantic_ai_agent()` helper works out of the box since `litellm` is already installed
- No separate `pydantic-ai-litellm` extra needed!

### Installation

```bash
# Install with Pydantic AI support
uv pip install graflow[pydantic-ai]

# Or with all features (includes pydantic-ai, adk, tracing, etc.)
uv pip install graflow[all]

# Note: LiteLLM is always available (core dependency)
# So create_pydantic_ai_agent() helper works with just [pydantic-ai]
```

### Environment Variables

**Provider API Keys** (auto-detected by Pydantic AI):
```bash
# OpenAI (for 'openai:*' or 'openai/*' models)
OPENAI_API_KEY=sk-...

# Anthropic (for 'anthropic:*' or 'anthropic/*' models)
ANTHROPIC_API_KEY=sk-ant-...

# Google (for 'gemini-*' models)
GOOGLE_API_KEY=...

# Other providers (as needed)
GROQ_API_KEY=...
MISTRAL_API_KEY=...
```

**Tracing** (required for Langfuse integration):
```bash
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com  # or https://us.cloud.langfuse.com
```

### Model Format Reference

**Direct Provider Syntax** (no LiteLLM):
```python
Agent('openai:gpt-4o')           # OpenAI
Agent('anthropic:claude-3-5-sonnet-20241022')  # Anthropic
Agent('gemini-1.5-pro')          # Google
```

**LiteLLM Syntax** (with `create_pydantic_ai_agent()` helper):
```python
# Helper uses LiteLLM backend with '/' separator
create_pydantic_ai_agent('openai/gpt-4o')
create_pydantic_ai_agent('anthropic/claude-3-5-sonnet-20241022')
create_pydantic_ai_agent('gemini-1.5-pro')
```

**Key Differences**:
- **Direct Agent**: Uses `:` separator (e.g., `Agent('openai:gpt-4o')`)
- **Helper Function**: Uses `/` separator (e.g., `create_pydantic_ai_agent('openai/gpt-4o')`)
- Both approaches produce valid Pydantic AI agents that can be wrapped with `PydanticLLMAgent`

---

## Summary

### Design Highlights

1. **Simple Wrapper**: Direct delegation to `pydantic_ai.Agent` (no abstraction layers)
2. **Type-Safe**: Leverages Pydantic AI's generic typing for structured outputs
3. **Multi-Provider**: Native support for OpenAI, Anthropic, Google, Groq, Mistral, etc.
4. **Pythonic**: Decorator-based tool registration, async/await patterns
5. **Lightweight**: Minimal dependencies compared to ADK
6. **Consistent**: Follows Graflow's LLMAgent interface and patterns
7. **Tracing**: Native Langfuse support via `Agent.instrument_all()`
8. **Flexible**: Optional LiteLLM backend for unified routing

### Key Advantages

- **Easier Setup**: No App/Runner/Session complexity (just `Agent(...)`)
- **Better Type Safety**: Full mypy/pyright support with generics
- **More Providers**: Built-in multi-provider support (OpenAI, Anthropic, Google, etc.)
- **Simpler Tools**: Decorator-based registration (`@agent.tool`)
- **Faster Development**: Rapid prototyping and iteration
- **Native Tracing**: Official Langfuse integration via OpenTelemetry
- **Flexible Agent Creation**: Users can create `Agent` instances any way they want, helper just provides LiteLLM convenience

### Design Patterns

**Two Usage Patterns**:

1. **Direct Provider** (simplest):
   ```python
   agent = Agent('openai:gpt-4o', instrument=True)
   wrapped = PydanticLLMAgent(agent, name="assistant")
   ```

2. **Using Helper** (LiteLLM backend for unified routing):
   ```python
   agent = create_pydantic_ai_agent('openai/gpt-4o')  # Helper uses LiteLLM
   wrapped = PydanticLLMAgent(agent, name="assistant")
   ```

**Tracing Setup** (once at startup):
```python
from graflow.llm.agents.pydantic_agent import setup_pydantic_ai_tracing

setup_pydantic_ai_tracing()  # Enables Langfuse tracing
```

### Next Steps

1. ✅ **Design Complete**: Design document finalized
2. **Implement Core**: Complete Phase 1 (PydanticLLMAgent class + setup_pydantic_ai_tracing)
3. **Implement Helper**: Add optional `create_pydantic_ai_agent()` helper (uses LiteLLM backend)
4. **Add Tests**: Unit and integration tests
5. **Create Examples**: Add to `examples/llm/`
6. **Documentation**: Update main LLM integration design doc

### Resolved Design Questions ✅

1. **Tracing Integration**: Langfuse official support via `Agent.instrument_all()`
2. **Helper Function**: Optional `create_pydantic_ai_agent()` helper uses LiteLLM backend
3. **Agent Construction**: User-supplied - can create `Agent` any way they want and wrap with `PydanticLLMAgent`

### Remaining Open Questions

See "Open Questions for Discussion" section above for:
- Dependencies type parameter support
- Model provider configuration patterns
- Agent serialization for distributed execution
- Tool state management in distributed workers
- Error handling strategy
- Usage limits integration

---

## References

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Pydantic AI Agents](https://ai.pydantic.dev/agents/)
- [Pydantic AI Models](https://ai.pydantic.dev/models/)
- [Graflow LLM Integration Design](./llm_integration_design.md)
- [Graflow ADK Agent Implementation](../../graflow/llm/agents/adk_agent.py)
