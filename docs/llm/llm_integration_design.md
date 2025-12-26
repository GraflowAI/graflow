# Graflow LLM Integration Design Document

## Overview

Design document for adding LLM integration features to Graflow using LiteLLM.

### Purpose

- Enable easy LLM usage within tasks
- Support multiple LLM providers (OpenAI, Anthropic, Google, etc.) through LiteLLM
- **Allow different models per task** (cost/performance optimization)
- **Unified tracing with LiteLLM's `langfuse_otel` and `ExecutionContext.trace_id`**
- Support Supervisor/ReAct patterns using Google ADK's LlmAgent

### Design Principles

1. **Loose Coupling**: LLM functionality is loosely coupled with Graflow's execution engine, serving as a utility within tasks
2. **Fat Single Node**: Supervisor/ReAct agents complete within a single task node, separate from Graflow's task graph
3. **Independence**: ADK tools and Graflow tasks are completely independent
4. **Dependencies**: LiteLLM and Google ADK are treated as optional dependencies
5. **Tracing**: Use LiteLLM's `langfuse_otel` callback, inheriting only `trace_id` (runs in parallel with Graflow tracer)
6. **Flexibility**: Use shared LLMClient instance with model override per `completion()` call

---

## Architecture

### Module Structure

```
graflow/llm/
├── __init__.py
├── client.py          # LLMClient - LiteLLM wrapper, setup_langfuse_for_litellm
├── serialization.py   # Agent serialization (YAML)
└── agents/
    ├── __init__.py
    ├── base.py        # LLMAgent - Base class for ReAct/Supervisor
    └── adk_agent.py   # AdkLLMAgent - Google ADK LlmAgent wrapper
```

**Note**: No custom tracing module needed as we use LiteLLM's `langfuse` callback and OpenTelemetry.

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│ Graflow Execution Layer                                       │
│                                                                │
│  ExecutionContext                                             │
│    ├─ trace_id (W3C TraceContext 32-digit hex) ←──┐          │
│    ├─ tracer (LangFuseTracer/NoopTracer)          │          │
│    └─ llm_client (LLMClient instance)              │          │
│                                                     │          │
│  Task Graph: [Task A] ──> [Supervisor Task] ──> [Task C]     │
│                                │                               │
│                                └─ Fat single node              │
│                                   (ReAct/Sub-agents internal)  │
└────────────────────────────────────────────────────────────────┘
                        │                            │
                        │ DI                         │ trace_id inheritance
                        ▼                            ▼
┌────────────────────────────────────────────────────────────────┐
│ LLM Layer (Loosely coupled with Graflow execution engine)     │
│                                                                │
│  LLMClient (completion + agent registry)                      │
│    ├─ completion(messages, **params)                          │
│    │   └─ Sets ExecutionContext.trace_id in                   │
│    │      metadata["trace_id"] when calling                   │
│    │      litellm.completion()                                │
│    ├─ register_agent(name, agent)                             │
│    └─ get_agent(name) -> LLMAgent                             │
│                                                                │
│  LLMAgent (base class for ReAct/Supervisor)                   │
│    └─ AdkLLMAgent (wraps ADK LlmAgent)                        │
│         ├─ sub_agents support                                 │
│         ├─ tools (independent, not graflow tasks)             │
│         └─ Calls LLMClient.completion() in run()              │
└────────────────────────────────────────────────────────────────┘
                        │
                        │ LiteLLM langfuse_otel callback
                        ▼
┌────────────────────────────────────────────────────────────────┐
│ Langfuse (via OpenTelemetry)                                  │
│                                                                │
│  Associates Graflow tasks and LLM calls with same trace_id    │
│  - Graflow tracer: Workflow/task traces                       │
│  - LiteLLM langfuse_otel: LLM call traces                     │
│  → Unified visualization in Langfuse UI                        │
└────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. LLMClient (`graflow/llm/client.py`)

LiteLLM wrapper providing:

1. **Completion API**: Simple access to LiteLLM's `completion()`
2. **Auto-tracing**: Auto-detects trace_id/span_id from OpenTelemetry context

#### Main Methods

```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from litellm import ModelResponse
else:
    ModelResponse = Any

class LLMClient:
    # Completion API
    def completion(
        messages,
        model=None,
        generation_name=None,
        tags=None,
        **params
    ) -> ModelResponse:
        """Returns LiteLLM's ModelResponse as-is"""

    def completion_text(messages, model=None, **params) -> str:
        """Convenience method: returns response.choices[0].message.content"""
```

#### Implementation Example

```python
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import importlib

if TYPE_CHECKING:
    from litellm import ModelResponse
else:
    ModelResponse = Any

class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        **default_params: Any
    ):
        self.model = model
        self.default_params = default_params

        try:
            self._litellm = importlib.import_module("litellm")
        except ImportError:
            raise RuntimeError("liteLLM is not installed.")

    def completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        generation_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **params: Any
    ) -> ModelResponse:
        """Unified completion API

        Args:
            messages: Message list
            model: Model override (optional)
            generation_name: Langfuse generation name (optional)
            tags: Langfuse tags (optional)
            **params: Additional LiteLLM parameters

        Returns:
            LiteLLM's ModelResponse object
            (access text via response.choices[0].message.content)
        """
        kwargs = {**self.default_params, **params}
        actual_model = model or self.model

        # Set Langfuse metadata
        if generation_name or tags:
            metadata = kwargs.get('metadata', {})
            if generation_name:
                metadata['generation_name'] = generation_name
            if tags:
                metadata['tags'] = tags
            kwargs['metadata'] = metadata

        # Auto-detect parent span from OpenTelemetry context
        return self._litellm.completion(
            model=actual_model,
            messages=messages,
            **kwargs
        )

    def completion_text(self, messages: List[Dict[str, str]], **params: Any) -> str:
        """Convenience method: returns response.choices[0].message.content"""
        response = self.completion(messages, **params)
        return extract_text(response)


def extract_text(response: ModelResponse) -> str:
    """Extract text from LiteLLM's ModelResponse"""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", None)
    return content or ""
```

#### Design Points

- **Unified API**: Provides all features via `completion()` (model, generation_name, tags)
- **Returns LiteLLM Response**: Thin wrapper returning LiteLLM's `ModelResponse` as-is
  - Avoids unnecessary abstraction, maintains access to all LiteLLM features
  - Standard access via `response.choices[0].message.content`
  - Easy text-only access via `completion_text()`
- **Model Override**: Temporarily change model with `completion(model="gpt-4")`
- **Langfuse Metadata**: Organize traces with `generation_name` and `tags`
- **Auto-tracing**: Auto-detects trace_id/span_id from OpenTelemetry context

---

### 2. LLMAgent (`graflow/llm/agents/base.py`)

Base class for ReAct/Supervisor patterns.

#### Main Methods

```python
from graflow.llm.agents.types import AgentResult

class LLMAgent(ABC):
    @abstractmethod
    def run(query: str, **kwargs) -> AgentResult:
        """Main agent logic (ReAct loop, sub-agent coordination, etc.)

        Returns:
            AgentResult (TypedDict):
                - output: Final agent output (str or Pydantic BaseModel if output_schema is configured)
                - steps: Execution trace (list of AgentStep dicts)
                - metadata: Additional metadata (dict)
        """
        pass

    def get_state() -> Dict[str, Any]:
        """Serialize state"""
        pass

    def set_state(state: Dict[str, Any]) -> None:
        """Restore state"""
        pass
```

#### Design Points

- **Fat Single Node**: Agent completes within a single task node
- **Independent Control Flow**: ReAct loops and sub-agent coordination implemented within `run()`
- **Separated from Graflow Graph**: Doesn't use dynamic task generation (`ctx.next_task()`)

---

### 3. AdkLLMAgent (`graflow/llm/agents/adk_agent.py`)

Wraps Google ADK's `LlmAgent` to conform to Graflow's `LLMAgent` interface.

#### Main Features

- **Sub-agents Support**: Leverages ADK's hierarchical agent structure
- **Tools Integration**: ADK's tool calling feature (independent of Graflow tasks)
- **Structured Output**: Supports `output_schema` for Pydantic BaseModel validation
- **LiteLLM Integration**: Uses LiteLLM with ADK

#### Implementation Example

```python
from google.adk.agents import LlmAgent
from graflow.llm.agents.types import AgentResult

class AdkLLMAgent(LLMAgent):
    def __init__(self, adk_agent: LlmAgent):
        """Wrap ADK LlmAgent"""
        self._adk_agent = adk_agent

    def run(self, input_text: str, **kwargs) -> AgentResult:
        """Execute ADK agent

        Returns:
            AgentResult with:
                - output: str or Pydantic BaseModel (if output_schema is configured)
                - steps: List of execution events
                - metadata: Agent metadata
        """
        adk_result = self._adk_agent.run(input_text, **kwargs)
        return self._convert_adk_result(adk_result)

# Usage example
from google.adk.agents import LlmAgent

# Create ADK LlmAgent
adk_agent = LlmAgent(
    name="supervisor",
    model="gemini-2.5-flash",
    tools=[search_tool, calculator_tool],
    sub_agents=[analyst_agent, writer_agent]
)

# Wrap for Graflow
agent = AdkLLMAgent(adk_agent)

# Register in ExecutionContext
context.register_llm_agent("supervisor", agent)

# Use in task
@task(inject_llm_agent="supervisor")
def supervise_task(agent: LLMAgent, query: str) -> str:
    result = agent.run(query)
    return result["output"]
```

---

### 4. LLM Configuration (Environment Variables)

LLM functionality configuration is loaded from environment variables (`.env` file).

**Environment Variables**:
```bash
# Langfuse tracing configuration
LANGFUSE_PUBLIC_KEY=pk-xxx    # Langfuse public API key
LANGFUSE_SECRET_KEY=sk-xxx    # Langfuse secret API key
LANGFUSE_HOST=https://cloud.langfuse.com  # Optional, default: cloud.langfuse.com

# Default LLM model
GRAFLOW_LLM_MODEL=gpt-5-mini  # Optional, default: gpt-5-mini
```

To enable tracing, call `setup_langfuse_for_litellm()`:

```python
from graflow.llm import setup_langfuse_for_litellm

# Load settings from .env and enable LiteLLM's Langfuse callback
setup_langfuse_for_litellm()
```

---

## ExecutionContext Integration

### ExecutionContext Extension

```python
class ExecutionContext:
    def __init__(
        self,
        # ... existing parameters ...
        llm_client: Optional[LLMClient] = None,
    ):
        self._llm_client = llm_client
        self._llm_agents: Dict[str, LLMAgent] = {}  # Agent Registry

    @property
    def llm_client(self) -> LLMClient:
        """Get LLM client (auto-creates with default model if None)"""
        if self._llm_client is None:
            # Lazy initialization with default model
            from graflow.utils.dotenv import load_env
            load_env()
            default_model = os.getenv("GRAFLOW_LLM_MODEL", "gpt-5-mini")
            self._llm_client = LLMClient(model=default_model)
        return self._llm_client

    def register_llm_agent(self, name: str, agent: LLMAgent) -> None:
        """Register LLMAgent"""
        self._llm_agents[name] = agent

    def get_llm_agent(self, name: str) -> LLMAgent:
        """Get LLMAgent"""
        if name not in self._llm_agents:
            raise KeyError(f"LLMAgent '{name}' not found in registry")
        return self._llm_agents[name]
```

### Factory Function

```python
def create_execution_context(
    # ... existing parameters ...
    llm_client: Optional[LLMClient] = None,
) -> ExecutionContext:
    """Create execution context (with LLM support)"""
    return ExecutionContext(
        # ... existing arguments ...
        llm_client=llm_client,
    )
```

### Usage Example

```python
from graflow.llm import LLMClient
from graflow.core.context import ExecutionContext

# Create and inject LLMClient directly
llm_client = LLMClient(
    model="gpt-5-mini",  # Default model (optional)
    temperature=0.7,
    max_tokens=1024
)

context = ExecutionContext.create(
    graph=graph,
    llm_client=llm_client  # Inject shared instance
)

# All tasks share the same LLMClient instance
# Can override model per task with completion(model="gpt-4o")
```

---

## Task Decorator Extension

### @task Decorator

```python
def task(
    id_or_func: Optional[F] | str | None = None,
    *,
    id: Optional[str] = None,
    inject_context: bool = False,
    inject_llm_client: bool = False,      # New
    inject_llm_agent: Optional[str] = None,  # New: specify agent name directly
    handler: Optional[str] = None
) -> TaskWrapper | Callable[[F], TaskWrapper]:
    """
    Task decorator

    Args:
        inject_context: Inject ExecutionContext as first argument
        inject_llm_client: Inject shared LLMClient instance as first argument
        inject_llm_agent: LLMAgent name (registered in ExecutionContext)
    """
```

### Usage Examples

```python
# Simple LLM task (uses default model)
@task(inject_llm_client=True)
def summarize(llm: LLMClient, text: str) -> str:
    """Uses LLMClient's default model (e.g., gpt-5-mini)"""
    messages = [
        {"role": "system", "content": "You are a summarization assistant."},
        {"role": "user", "content": f"Summarize: {text}"}
    ]
    return llm.completion_text(messages)

# Override model per task
@task(inject_llm_client=True)
def complex_analysis(llm: LLMClient, data: str) -> str:
    """
    Can override model per completion() call

    - All tasks share the same LLMClient instance
    - Change model per task as needed
    - Uses shared default_params for temperature, etc.
    """
    # High-performance model for analysis
    response = llm.completion(
        [{"role": "user", "content": data}],
        model="gpt-4o"  # Use gpt-4o for this task only
    )
    return response.choices[0].message.content

# Use multiple models
@task(inject_llm_client=True)
def multi_model_task(llm: LLMClient, text: str) -> dict:
    """Use multiple models within a task"""

    # Simple tasks with low-cost model
    summary = llm.completion_text(
        [{"role": "user", "content": f"Summarize: {text}"}],
        model="gpt-5-mini"
    )

    # Complex reasoning with high-performance model
    analysis = llm.completion_text(
        [{"role": "user", "content": f"Analyze deeply: {text}"}],
        model="claude-3-5-sonnet-20241022"
    )

    return {"summary": summary, "analysis": analysis}

# Use LLM Agent
@task(inject_llm_agent="supervisor")
def run_supervisor(agent: LLMAgent, query: str) -> str:
    """Get 'supervisor' from Agent Registry and execute"""
    return agent.run(query)
```

**Important**: New design features:
- All tasks share **the same LLMClient instance** (memory efficient)
- Override model per `completion(model="...")` call
- Select optimal model per task/call
- Shared default_params (temperature, etc.)

### TaskWrapper Implementation

```python
class TaskWrapper(Executable):
    def __init__(
        self,
        task_id: str,
        func: Callable,
        inject_context: bool = False,
        inject_llm_client: bool = False,
        inject_llm_agent: Optional[str] = None,
        handler_type: Optional[str] = None,
    ):
        self.inject_llm_client = inject_llm_client
        self.inject_llm_agent = inject_llm_agent
        # ...

    def __call__(self, *args, **kwargs) -> Any:
        exec_context = self.get_execution_context()

        # LLMClient injection (shared instance)
        if self.inject_llm_client:
            llm_client = exec_context.llm_client

            # Always available (auto-created if needed)
            # Inject shared LLMClient instance
            # Can override model with completion(model="...") in task
            return self.func(llm_client, *args, **kwargs)

        # LLMAgent injection
        if self.inject_llm_agent:
            try:
                agent = exec_context.get_llm_agent(self.inject_llm_agent)
            except KeyError:
                raise RuntimeError(
                    f"Task {self.task_id} requires LLMAgent '{self.inject_llm_agent}' "
                    "but not found in registry. Use ctx.register_llm_agent() first."
                )

            return self.func(agent, *args, **kwargs)

        # ... other injection logic ...
```

---

## Tracing Integration

### Auto-propagation via OpenTelemetry

Associate Graflow workflows and LLM calls automatically through **OpenTelemetry's current context**.

#### Tracing Architecture

```
LangFuseTracer (Graflow)
    │
    ├─ span_start("task_a")
    │   └─ Set SpanContext in OpenTelemetry context
    │       ↓
    │   [Current Context with trace_id + span_id]
    │
    ├─ Task execution
    │   ├─ LiteLLM completion()
    │   │   └─ Auto-retrieved from Current Context ✅
    │   │       → Sent to Langfuse as child span
    │   │
    │   └─ ADK agent.run()
    │       └─ Auto-retrieved from Current Context ✅
    │           → Sent to Langfuse as child span
    │           ├─ sub_agent calls
    │           └─ tool calls
    │
    └─ span_end("task_a")
        └─ Clear OpenTelemetry context
```

#### Tracing Paths

1. **Graflow Tasks**: `LangFuseTracer` in `graflow/trace/langfuse.py`
   - Workflow start/end, task execution (recorded as spans)
   - **Sets OpenTelemetry context** ← New addition
   - Parallel groups, dynamic task generation, etc.

2. **LiteLLM**: LiteLLM's built-in Langfuse integration
   - Enabled with `litellm.callbacks = ["langfuse"]`
   - **Auto-detects parent span from OpenTelemetry context**
   - Model name, prompts, responses, token counts, etc.

3. **Google ADK** (Optional): ADK's built-in Langfuse integration
   - Enabled with `GoogleADKInstrumentor().instrument()`
   - **Auto-detects parent span from OpenTelemetry context**
   - ReAct loops, sub-agents, tool calls, etc.

---

## Usage Examples

### 1. Simple LLM Task

```python
from graflow.core.decorators import task
from graflow.llm import LLMClient, setup_langfuse_for_litellm

# Enable tracing (loads settings from .env)
setup_langfuse_for_litellm()

@task(inject_llm_client=True)
def summarize(llm: LLMClient, text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a summarization assistant."},
        {"role": "user", "content": f"Summarize: {text}"}
    ]
    return llm.completion_text(messages)

# Execute
result = summarize.run(text="Long article...")
```

### 2. LLM Task in Workflow

```python
from graflow.core.workflow import workflow
from graflow.llm import LLMClient, setup_langfuse_for_litellm

# Enable tracing
setup_langfuse_for_litellm()

# Create LLMClient
llm_client = LLMClient(model="gpt-5-mini")

@task(inject_llm_client=True)
def analyze_sentiment(llm: LLMClient, text: str) -> str:
    messages = [{"role": "user", "content": f"Sentiment of: {text}"}]
    return llm.completion_text(messages)

@task(inject_llm_client=True)
def extract_entities(llm: LLMClient, text: str) -> List[str]:
    messages = [{"role": "user", "content": f"Extract entities: {text}"}]
    result = llm.completion_text(messages)
    return result.split(", ")

@task
def combine_results(sentiment: str, entities: List[str]) -> Dict:
    return {"sentiment": sentiment, "entities": entities}

# Workflow
with workflow("nlp_pipeline") as wf:
    text = "Apple Inc. announced great earnings today."

    sentiment = analyze_sentiment(text=text)
    entities = extract_entities(text=text)
    result = combine_results(sentiment, entities)

    sentiment >> result
    entities >> result

    wf.execute()
```

---

## Design Key Points

### 1. Loose Coupling Architecture

- **LLM functionality is loosely coupled with Graflow execution engine**
  - LLMClient/LLMAgent are utilities injected via DI
  - Independent of Graflow's task execution logic
  - Built into ExecutionContext but optional

### 2. Fat Single Node Design

- **Supervisor/ReAct completes within a single task node**
  - Doesn't use dynamic task generation (`ctx.next_task()`)
  - ADK sub_agents and tool calls handled internally
  - Graflow graph focuses on coarse-grained workflow management

### 3. Tools and Tasks Separation

- **ADK tools ≠ Graflow tasks**
  - ADK tools are independent Python functions
  - Graflow tasks are workflow components
  - Don't integrate them (avoid confusion)

### 4. Model Selection Flexibility

- **Two-level model specification**
  1. LLMClient instance level: `LLMClient(model="...")`
  2. Runtime level: `llm.completion(..., model="...")`

### 5. Tracing Integration

- **Auto-propagates trace_id/session_id**
  - LLMClient inherits ExecutionContext's trace_id
  - Associates Graflow workflows and LLM calls in Langfuse
  - Easy tracing activation via environment variables

---

## Dependencies

### Required

- `python >= 3.11`
- `graflow` (core)

### Optional

- `litellm >= 1.0.0` - LLM provider integration (OpenAI, Anthropic, Google, etc.)
  - Includes Langfuse integration
- `google-adk >= 0.1.0` - Google ADK (Supervisor/ReAct patterns)
  - Includes LiteLLM support

**Note**: OpenTelemetry not required. LiteLLM and ADK provide built-in Langfuse integration.

### pyproject.toml

```toml
[project.optional-dependencies]
llm = [
    "litellm>=1.0.0",
]
adk = [
    "google-adk>=0.1.0",
]
# Enable all LLM features
all-llm = [
    "litellm>=1.0.0",
    "google-adk>=0.1.0",
]
```

### Installation

```bash
# LiteLLM only
uv pip install graflow[llm]

# Including ADK
uv pip install graflow[all-llm]

# Or
pip install "graflow[all-llm]"
```

---

## FAQ

### Q1: Can different models be used per task?

**A**: Yes, override the model per `completion()` call:

```python
# Create LLMClient (default model optional)
llm_client = LLMClient(
    model="gpt-5-mini",  # Default model (optional)
    temperature=0.7,
    max_tokens=1024
)
context = ExecutionContext.create(graph, llm_client=llm_client)

# Task A: Use default model (gpt-5-mini)
@task(inject_llm_client=True)
def summarize(llm: LLMClient, text: str) -> str:
    # Uses default model
    return llm.completion_text([{"role": "user", "content": text}])

# Task B: Use high-performance model (gpt-4o)
@task(inject_llm_client=True)
def analyze(llm: LLMClient, data: str) -> str:
    # Override model per completion() call
    return llm.completion_text(
        [{"role": "user", "content": data}],
        model="gpt-4o"  # Use gpt-4o for this task only
    )

# Task C: Use different provider (Claude)
@task(inject_llm_client=True)
def reason(llm: LLMClient, problem: str) -> str:
    # Override model per completion() call
    return llm.completion_text(
        [{"role": "user", "content": problem}],
        model="claude-3-5-sonnet-20241022"
    )

# Task D: Use multiple models within same task
@task(inject_llm_client=True)
def multi_model(llm: LLMClient, text: str) -> dict:
    # Simple summary with low-cost model
    summary = llm.completion_text(
        [{"role": "user", "content": f"Summarize: {text}"}],
        model="gpt-5-mini"
    )

    # Complex analysis with high-performance model
    analysis = llm.completion_text(
        [{"role": "user", "content": f"Analyze: {text}"}],
        model="claude-3-5-sonnet-20241022"
    )

    return {"summary": summary, "analysis": analysis}
```

**Key Points**:
- All tasks share **the same LLMClient instance** (memory efficient)
- Override model per `completion(model="...")` call per task/invocation
- `default_params` (temperature, max_tokens, etc.) are shared
- Can use multiple models within the same task

**Use Cases**:
- Simple tasks: `gpt-5-mini` (low cost)
- Complex analysis: `gpt-4o` or `claude-3-5-sonnet-20241022` (high performance)
- Code generation: `claude-3-5-sonnet-20241022` (code-specialized)

### Q2: Can `inject_context` and `inject_llm_client` be used together?

**A**: Yes:

```python
# Use both (e.g., to access Agent Registry)
@task(inject_context=True, inject_llm_client=True)
def my_task(ctx: ExecutionContext, llm: LLMClient, data: str):
    # LLM completion
    result = llm.completion_text([...])

    # Register agent
    agent = create_agent()
    ctx.register_llm_agent("my_agent", agent)

    return result

# Or access via ctx
@task(inject_context=True)
def my_task(ctx: ExecutionContext, data: str):
    llm = ctx.llm_client
    llm.completion_text([...])
```

### Q3: What happens to LLMClient in distributed execution?

**A**: LLMClient is serialized and reconstructed in Worker:

1. ExecutionContext is serialized to Worker
2. Worker restores ExecutionContext
3. Worker loads environment variables from .env, restoring tracing settings
4. Lazy initialization when `llm_client` property is accessed
5. Langfuse tracing is automatically enabled in Worker

### Q4: What happens to LLMAgent in distributed execution?

**A**: Agents are serialized as YAML and sent to Worker:

```python
# Agent serialization
from google.adk.utils.yaml_utils import dump_pydantic_to_yaml
import io

def agent_to_yaml(agent: BaseAgent) -> str:
    """Convert BaseAgent to YAML string"""
    buf = io.StringIO()
    dump_pydantic_to_yaml(agent, buf)
    return buf.getvalue()

# Agent restoration
import yaml
from google.adk.agents import BaseAgent

def yaml_to_agent(yaml_str: str) -> BaseAgent:
    """Restore BaseAgent from YAML string"""
    cfg = yaml.safe_load(yaml_str)
    agent = BaseAgent.from_config(cfg)
    return agent
```

**ExecutionContext handling:**

1. Convert Agent to YAML during `register_llm_agent()`
2. Include YAML string during Worker serialization
3. Restore from YAML during `get_llm_agent()` in Worker

---

## Summary

### Design Core

This design achieves simple yet powerful LLM integration by leveraging **OpenTelemetry's auto-propagation**.

#### Three Integration Layers

1. **Graflow Layer**
   - DI LLMClient with `@task(inject_llm_client=True)`
   - `ExecutionContext` manages LLMClient lifecycle
   - `LangFuseTracer` sets OpenTelemetry context

2. **LLM Layer**
   - `LLMClient`: Simple LiteLLM wrapper (completion API)
   - `LLMAgent`: Google ADK wrapper (Supervisor/ReAct patterns)
   - Agent Registry: Managed in ExecutionContext (simple dict)

3. **Tracing Layer**
   - LangFuseTracer sets trace_id/span_id in OpenTelemetry context
   - LiteLLM auto-detects parent span from current context
   - Unified visualization in Langfuse UI (no manual trace_id passing)

#### Auto-propagation Benefits

**No manual management:**
```python
# ✅ Simple and robust
llm = LLMClient(model="gpt-5-mini")
llm.completion(messages)  # trace_id/span_id auto-detected!
```

**Implementation simplicity:**
- **LLMClient**: No trace_id/span_id parameters needed
- **LangFuseTracer**: ~10 lines added to `_output_span_start()` and `_output_span_end()`
- **LiteLLM**: Uses existing Langfuse callback (no changes)

#### Tracing Visualization

Langfuse UI display example:

```
Trace: workflow_execution (trace_id: abc123...)
  └─ Span: supervisor_task
      ├─ Span: litellm.completion (model: gpt-5-mini)  ← Auto-hierarchy
      │   └─ usage: {total_tokens: 150}
      └─ Span: adk.agent.run (agent: supervisor)       ← Auto-hierarchy
          ├─ Span: sub_agent.researcher
          │   └─ Span: tool.web_search
          └─ Span: sub_agent.writer
```

### Success Criteria

- ✅ Easy LLM usage in tasks with `@task(inject_llm_client=True)`
- ✅ Easy Agent usage in tasks with `@task(inject_llm_agent=True, agent_name="...")`
- ✅ Agent Registry managed in ExecutionContext (simple dict)
- ✅ Support multiple LLM providers via LiteLLM (OpenAI, Anthropic, Google, etc.)
- ✅ Unified tracing of Graflow tasks and LLM calls in Langfuse
- ✅ Support Supervisor/ReAct patterns with Google ADK
- ✅ Loose coupling architecture independent of Graflow execution engine
- ✅ No manual trace_id/span_id management with OpenTelemetry auto-propagation

---

## References

- [LiteLLM Documentation](https://docs.litellm.ai/)
- [LiteLLM Langfuse Integration](https://docs.litellm.ai/docs/observability/langfuse_integration)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Google ADK Documentation](https://developers.google.com/adk)
- Graflow existing design documents:
  - `docs/architecture_ja.md`
  - `docs/trace_module_design_ja.md`
