# LLM Integration Examples

Examples demonstrating LLM integration features in Graflow workflows.

## Overview

Graflow provides seamless LLM integration through:
- **LLMClient**: Direct LLM API access via LiteLLM (supports OpenAI, Anthropic, Google, etc.)
- **LLMAgent**: ReAct/Supervisor patterns using Google ADK's LlmAgent
- **Dependency Injection**: Automatic injection of LLM clients and agents into tasks
- **Model Override**: Per-task model selection for cost/performance optimization
- **Tracing**: Unified tracing with Langfuse integration

## Prerequisites

### Basic Setup (LLMClient)
```bash
# Install LiteLLM support
uv add litellm

# Set API key in .env
OPENAI_API_KEY=sk-...

# Optional: Set default model
GRAFLOW_LLM_MODEL=gpt-5-mini
```

### Advanced Setup (LLMAgent)
```bash
# Install Google ADK
uv add google-adk

# Set Google API key
GOOGLE_API_KEY=...
```

### Optional: Tracing Setup
```bash
# Set Langfuse credentials for tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Examples

### 1. Simple LLMClient (`simple_llm_client.py`)

**Concepts**: Basic LLMClient injection, automatic initialization, shared instance

**What it demonstrates**:
- Using `@task(inject_llm_client=True)` for automatic LLMClient injection
- Auto-creation from environment variables
- Basic completion() calls with LiteLLM
- Sequential LLM-powered workflow

**Run**:
```bash
PYTHONPATH=. uv run python examples/12_llm_integration/simple_llm_client.py
```

**Key code**:
```python
@task(inject_llm_client=True)
def my_task(llm):
    response = llm.completion(
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=50
    )
    return response.choices[0].message.content
```

### 2. Model Override (`model_override.py`)

**Concepts**: Per-task model selection, cost optimization, multi-provider workflows

**What it demonstrates**:
- Overriding model per completion() call
- Cost optimization strategy (cheap vs expensive models)
- Using different models for different task complexities
- Shared LLMClient with model flexibility

**Run**:
```bash
PYTHONPATH=. uv run python examples/12_llm_integration/model_override.py
```

**Key code**:
```python
@task(inject_llm_client=True)
def simple_task(llm):
    # Use cheap model for simple tasks
    return llm.completion(model="gpt-5-mini", messages=[...])

@task(inject_llm_client=True)
def complex_task(llm):
    # Use powerful model for complex reasoning
    return llm.completion(model="gpt-4o", messages=[...])
```

### 3. LLMAgent with Google ADK (`llm_agent.py`)

**Concepts**: LLMAgent injection, tool definition, ReAct pattern

**What it demonstrates**:
- Registering LLMAgent in ExecutionContext
- Using `@task(inject_llm_agent="agent_name")` for agent injection
- Defining tools for agent use
- Google ADK LlmAgent for complex reasoning

**Run**:
```bash
PYTHONPATH=. uv run python examples/12_llm_integration/llm_agent.py
```

**Key code**:
```python
# Define tools
def my_tool(param: str) -> str:
    """Tool description."""
    return result

# Create and register agent
from google_adk.types import LlmAgent
from graflow.llm.agents.adk_agent import AdkLLMAgent

adk_agent = LlmAgent(name="assistant", model="gemini-2.0-flash-exp", tools=[my_tool])
agent = AdkLLMAgent(adk_agent, app_name=exec_context.session_id)
exec_context.register_llm_agent("assistant", agent)

# Use in task
@task(inject_llm_agent="assistant")
def my_task(agent):
    return agent.send_message("Use my_tool to...")
```

### 4. Multi-Agent Workflow (`multi_agent_workflow.py`)

**Concepts**: Multiple specialized agents, agent collaboration, mixing LLMClient and LLMAgent

**What it demonstrates**:
- Registering multiple agents with different tools
- Specialized agents (Researcher, Analyst, Writer)
- Agent collaboration via task dependencies
- Combining LLMClient and LLMAgent in same workflow
- Data passing between agents

**Run**:
```bash
PYTHONPATH=. uv run python examples/12_llm_integration/multi_agent_workflow.py
```

**Key code**:
```python
# Register specialized agents
exec_context.register_llm_agent("researcher", researcher_agent)
exec_context.register_llm_agent("analyst", analyst_agent)
exec_context.register_llm_agent("writer", writer_agent)

# Define collaborative workflow
@task(inject_llm_agent="researcher")
def research(agent):
    return agent.send_message("Research...")

@task(inject_llm_agent="analyst")
def analyze(agent):
    return agent.send_message("Analyze...")

@task(inject_llm_agent="writer")
def write(agent):
    return agent.send_message("Write...")

@task(inject_llm_client=True)
def review(llm):
    return llm.completion(messages=[...])

# Pipeline
research >> analyze >> write >> review
```

## Quick Start Guide

### Step 1: Basic LLM Task
```python
from graflow.core.decorators import task
from graflow.core.workflow import workflow

with workflow("my_workflow") as ctx:
    @task(inject_llm_client=True)
    def llm_task(llm):
        response = llm.completion(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        return response.choices[0].message.content

    ctx.execute("llm_task")
```

### Step 2: Model Override
```python
@task(inject_llm_client=True)
def smart_task(llm):
    # Use cheap model first
    summary = llm.completion(model="gpt-5-mini", messages=[...])

    # Use expensive model for complex reasoning
    analysis = llm.completion(model="gpt-4o", messages=[...])

    return analysis
```

### Step 3: Add an Agent
```python
from google_adk.types import LlmAgent
from graflow.llm.agents.adk_agent import AdkLLMAgent

def my_tool(input: str) -> str:
    """Custom tool."""
    return f"Processed: {input}"

@task(inject_context=True)
def setup(context):
    exec_context = context.execution_context
    adk_agent = LlmAgent(name="assistant", model="gemini-2.0-flash-exp", tools=[my_tool])
    agent = AdkLLMAgent(adk_agent, app_name=exec_context.session_id)
    exec_context.register_llm_agent("assistant", agent)

@task(inject_llm_agent="assistant")
def agent_task(agent):
    return agent.send_message("Use my_tool to process 'hello'")

setup >> agent_task
```

## Common Patterns

### Pattern 1: Cost Optimization
```python
# Use cheap models for simple tasks
@task(inject_llm_client=True)
def classify(llm):
    return llm.completion(model="gpt-5-mini", messages=[...])

# Use expensive models for complex tasks
@task(inject_llm_client=True)
def deep_analysis(llm):
    return llm.completion(model="gpt-4o", messages=[...])
```

### Pattern 2: Multi-Provider
```python
# OpenAI for code
@task(inject_llm_client=True)
def generate_code(llm):
    return llm.completion(model="gpt-4o", messages=[...])

# Anthropic for reasoning
@task(inject_llm_client=True)
def reason(llm):
    return llm.completion(model="claude-3-5-sonnet-20241022", messages=[...])

# Google for speed
@task(inject_llm_client=True)
def quick_task(llm):
    return llm.completion(model="gemini-2.0-flash-exp", messages=[...])
```

### Pattern 3: Agent Specialization
```python
# Specialized agents with different tools
researcher = LlmAgent(name="researcher", tools=[fetch_data, search_web])
analyst = LlmAgent(name="analyst", tools=[calculate, visualize])
writer = LlmAgent(name="writer", tools=[format_text, check_grammar])

# Register all agents
exec_context.register_llm_agent("researcher", AdkLLMAgent(researcher, app_name=...))
exec_context.register_llm_agent("analyst", AdkLLMAgent(analyst, app_name=...))
exec_context.register_llm_agent("writer", AdkLLMAgent(writer, app_name=...))

# Use in pipeline
research_task >> analysis_task >> writing_task
```

## Model Recommendations

### For Cost-Conscious Workflows
- **gpt-5-mini** (OpenAI): Fast, cheap, good for simple tasks
- **claude-3-5-haiku** (Anthropic): Fast, affordable
- **gemini-2.0-flash-exp** (Google): Very fast, free tier available

### For Quality-Critical Workflows
- **gpt-4o** (OpenAI): Strong reasoning, code generation
- **claude-3-5-sonnet-20241022** (Anthropic): Best reasoning, long context
- **gemini-2.0-flash-thinking-exp** (Google): Extended thinking, complex reasoning

### For Agent-Based Workflows
- **gemini-2.0-flash-exp** (Google): Best tool use, fast
- **gpt-4o** (OpenAI): Good function calling
- **claude-3-5-sonnet-20241022** (Anthropic): Strong reasoning with tools

## Troubleshooting

### Issue: "API key not found"
**Solution**: Set API key in .env file:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### Issue: "Module 'google_adk' not found"
**Solution**: Install Google ADK:
```bash
uv add google-adk
```

### Issue: "Model not found"
**Solution**: Check LiteLLM docs for supported models or verify API key is correct for the provider.

### Issue: Agent tools not working
**Solution**: Ensure tools have proper docstrings with Args/Returns sections. Google ADK uses these for tool descriptions.

## Next Steps

1. **Start Simple**: Begin with `simple_llm_client.py` to understand basic injection
2. **Optimize Costs**: Use `model_override.py` patterns for production workflows
3. **Add Agents**: Try `llm_agent.py` when you need tool use and complex reasoning
4. **Scale Up**: Build multi-agent systems with `multi_agent_workflow.py` patterns

## Additional Resources

- [LLM Integration Design Doc](../../docs/llm_integration_design.md)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Google ADK Documentation](https://ai.google.dev/gemini-api/docs/agent-development-kit)
- [Langfuse Documentation](https://langfuse.com/docs)
