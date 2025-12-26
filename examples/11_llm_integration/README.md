# LLM Integration Examples

Examples demonstrating LLM integration features in Graflow workflows.

## Overview

Graflow provides seamless LLM integration through:
- **LLMClient**: Direct LLM API access via LiteLLM (supports OpenAI, Anthropic, Google, etc.)
- **LLMAgent**: ReAct/Supervisor patterns using Google ADK's LlmAgent
- **Dependency Injection**: Automatic injection of LLM clients and agents into tasks
- **Model Override**: Per-task model selection for cost/performance optimization
- **Tracing**: Unified tracing with Langfuse integration

## Setup

From the repository root, install the example dependencies with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r examples/11_llm_integration/requirements.txt
```

Or run scripts directly with a temporary environment:

```bash
uv run --with examples/11_llm_integration/requirements.txt python examples/11_llm_integration/your_script.py
```

> Replace `your_script.py` with any of the example files in this directory (e.g., `simple_llm_client.py`).

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

**Option 1: Use Langfuse Cloud**
```bash
# Set Langfuse credentials for tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Option 2: Run Langfuse Locally with Docker**

For local development and testing, you can run your own Langfuse server:

```bash
# Get a copy of the latest Langfuse repository
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# Run the Langfuse docker compose
docker compose up
```

Once running, configure your environment to use the local instance:
```bash
LANGFUSE_PUBLIC_KEY=<get-from-local-ui>
LANGFUSE_SECRET_KEY=<get-from-local-ui>
LANGFUSE_HOST=http://localhost:3000
```

**Note**: If your application runs inside Docker, use `LANGFUSE_HOST=http://host.docker.internal:3000` to access the Langfuse server running on your host machine.

Visit `http://localhost:3000` to access the Langfuse UI and create your API keys.

For more information about Langfuse, visit the [Langfuse GitHub repository](https://github.com/langfuse/langfuse).

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
uv run python examples/11_llm_integration/simple_llm_client.py
```

**Key code**:
```python
@task(inject_llm_client=True)
def my_task(llm_client):
    response = llm_client.completion(
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
uv run python examples/11_llm_integration/model_override.py
```

**Key code**:
```python
@task(inject_llm_client=True)
def simple_task(llm_client):
    # Use cheap model for simple tasks
    return llm_client.completion(model="gpt-5-mini", messages=[...])

@task(inject_llm_client=True)
def complex_task(llm_client):
    # Use powerful model for complex reasoning
    return llm_client.completion(model="gpt-4o", messages=[...])
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
uv run python examples/11_llm_integration/llm_agent.py
```

**Key code**:
```python
# Define tools
def my_tool(param: str) -> str:
    """Tool description."""
    return result

# Create and register agent (two patterns)
from google.adk.agents import LlmAgent
from google.adk.apps import App
from graflow.llm.agents.adk_agent import AdkLLMAgent

# Pattern 1: Pass LlmAgent with app_name (simpler)
adk_agent = LlmAgent(name="assistant", model="gemini-2.5-flash", tools=[my_tool])
agent = AdkLLMAgent(adk_agent, app_name=exec_context.session_id)

# Pattern 2: Create and pass App (recommended, more control)
app = App(name=exec_context.session_id, root_agent=adk_agent)
agent = AdkLLMAgent(app)  # Optional: user_id="custom-user"

exec_context.register_llm_agent("assistant", agent)

# Use in task
@task(inject_llm_agent="assistant")
def my_task(llm_agent):
    result = llm_agent.run("Use my_tool to...")
    return result["output"]
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
uv run python examples/11_llm_integration/multi_agent_workflow.py
```

**Key code**:
```python
# Register specialized agents
exec_context.register_llm_agent("researcher", researcher_agent)
exec_context.register_llm_agent("analyst", analyst_agent)
exec_context.register_llm_agent("writer", writer_agent)

# Define collaborative workflow
@task(inject_llm_agent="researcher")
def research(llm_agent):
    return llm_agent.run("Research...")["output"]

@task(inject_llm_agent="analyst")
def analyze(llm_agent):
    return llm_agent.run("Analyze...")["output"]

@task(inject_llm_agent="writer")
def write(llm_agent):
    return llm_agent.run("Write...")["output"]

@task(inject_llm_client=True)
def review(llm_client):
    return llm_client.completion(messages=[...])

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
    def llm_task(llm_client):
        response = llm_client.completion(
            messages=[{"role": "user", "content": "Hello!"}]
        )
        return response.choices[0].message.content

    ctx.execute("llm_task")
```

### Step 2: Model Override
```python
@task(inject_llm_client=True)
def smart_task(llm_client):
    # Use cheap model first
    summary = llm_client.completion(model="gpt-5-mini", messages=[...])

    # Use expensive model for complex reasoning
    analysis = llm_client.completion(model="gpt-4o", messages=[...])

    return analysis
```

### Step 3: Add an Agent
```python
from google.adk.agents import LlmAgent
from google.adk.apps import App
from graflow.llm.agents.adk_agent import AdkLLMAgent

def my_tool(input: str) -> str:
    """Custom tool."""
    return f"Processed: {input}"

with workflow("agent_workflow") as ctx:
    def create_agent(exec_context):
        adk_agent = LlmAgent(name="assistant", model="gemini-2.5-flash", tools=[my_tool])

        # Pattern 1: Pass LlmAgent with app_name
        return AdkLLMAgent(adk_agent, app_name=exec_context.session_id)

        # Pattern 2: Create and pass App (recommended)
        # app = App(name=exec_context.session_id, root_agent=adk_agent)
        # return AdkLLMAgent(app)

    ctx.register_llm_agent("assistant", create_agent)

    @task(inject_llm_agent="assistant")
    def agent_task(llm_agent):
        return llm_agent.run("Use my_tool to process 'hello'")["output"]

    ctx.execute("agent_task")
```

## Common Patterns

### Pattern 1: Cost Optimization
```python
# Use cheap models for simple tasks
@task(inject_llm_client=True)
def classify(llm_client):
    return llm_client.completion(model="gpt-5-mini", messages=[...])

# Use expensive models for complex tasks
@task(inject_llm_client=True)
def deep_analysis(llm_client):
    return llm_client.completion(model="gpt-4o", messages=[...])
```

### Pattern 2: Multi-Provider
```python
# OpenAI for code
@task(inject_llm_client=True)
def generate_code(llm_client):
    return llm_client.completion(model="gpt-4o", messages=[...])

# Anthropic for reasoning
@task(inject_llm_client=True)
def reason(llm_client):
    return llm_client.completion(model="claude-3-5-sonnet-20241022", messages=[...])

# Google for speed
@task(inject_llm_client=True)
def quick_task(llm_client):
    return llm_client.completion(model="gemini-2.5-flash", messages=[...])
```

### Pattern 3: Agent Specialization
```python
# Specialized agents with different tools
researcher = LlmAgent(name="researcher", tools=[fetch_data, search_web])
analyst = LlmAgent(name="analyst", tools=[calculate, visualize])
writer = LlmAgent(name="writer", tools=[format_text, check_grammar])

# Register all agents when the workflow executes
ctx.register_llm_agent("researcher", lambda exec_ctx: AdkLLMAgent(researcher, app_name=exec_ctx.session_id))
ctx.register_llm_agent("analyst", lambda exec_ctx: AdkLLMAgent(analyst, app_name=exec_ctx.session_id))
ctx.register_llm_agent("writer", lambda exec_ctx: AdkLLMAgent(writer, app_name=exec_ctx.session_id))

# Use in pipeline
research_task >> analysis_task >> writing_task
```

## Model Recommendations

### For Cost-Conscious Workflows
- **gpt-5-mini** (OpenAI): Fast, cheap, good for simple tasks
- **claude-3-5-haiku** (Anthropic): Fast, affordable
- **gemini-2.5-flash** (Google): Very fast, free tier available

### For Quality-Critical Workflows
- **gpt-4o** (OpenAI): Strong reasoning, code generation
- **claude-3-5-sonnet-20241022** (Anthropic): Best reasoning, long context
- **gemini-2.5-flash** (Google): Extended thinking, complex reasoning

### For Agent-Based Workflows
- **gemini-2.5-flash** (Google): Best tool use, fast
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

### Issue: "App name mismatch detected" warning
**Solution**: This warning is harmless and can be safely ignored. It occurs because ADK infers app_name from the import path (`google.adk.agents` → "agents"), while Graflow uses session_id for proper workflow identification. The warning doesn't affect functionality.

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
