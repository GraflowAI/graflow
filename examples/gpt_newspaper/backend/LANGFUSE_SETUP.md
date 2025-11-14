# LangFuse Tracing Setup Guide

## Overview

Graflow's LangFuse integration uses **OpenTelemetry context propagation** to automatically link:
- Graflow workflow/task traces (via `LangFuseTracer`)
- LLM API calls (via LiteLLM's Langfuse callback)

**No manual trace ID passing needed** - it's all automatic!

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `langfuse>=3.8.1` - Langfuse SDK (3.8.1+ required for LiteLLM compatibility)
- `opentelemetry-api>=1.37.0` - OpenTelemetry context API
- `opentelemetry-sdk>=1.37.0` - OpenTelemetry SDK
- `opentelemetry-exporter-otlp>=1.37.0` - OTLP exporter (required for langfuse_otel callback)
- `litellm>=1.72.6` - LLM API wrapper (1.72.6+ required for langfuse_otel callback)

**Important**: Langfuse version 3.8.1 or higher is required for the OpenTelemetry-based integration. If you see errors like `TypeError: Langfuse.__init__() got an unexpected keyword argument 'sdk_integration'`, ensure you have:

1. Upgraded to langfuse 3.8.1+:
   ```bash
   pip install --upgrade 'langfuse>=3.8.1'
   ```

2. The code is using `langfuse_otel` callback (not the legacy `langfuse` callback)

**Note**: The `langfuse_otel` callback was introduced in LiteLLM v1.72.6+ and is required for compatibility with Langfuse SDK v3+. The older `langfuse` callback only works with Langfuse SDK v2 and will cause errors with v3.

### 2. Set Environment Variables

Add to your `.env` file:

```bash
# Langfuse credentials (required for tracing)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Optional: Custom Langfuse host (default: https://cloud.langfuse.com)
LANGFUSE_HOST=https://cloud.langfuse.com

# Or for local Langfuse instance:
# LANGFUSE_HOST=http://localhost:3000
```

Get your keys at: https://cloud.langfuse.com/

## How It Works

### Architecture

```
1. LangFuseTracer (Graflow)
   └─ Sets OpenTelemetry context (trace_id, span_id)
      ↓
2. LLMClient.completion()
   └─ LiteLLM auto-detects OpenTelemetry context via Langfuse callback
      ↓
3. LiteLLM Langfuse integration
   └─ Creates nested spans in Langfuse
```

### Code Flow

```python
# 1. Langfuse is automatically enabled when LLMClient is initialized
# The first time a task accesses context.llm_client, the following happens:
#   - LLMClient.__init__() is called with enable_tracing=True (default)
#   - setup_langfuse_for_litellm() is automatically called
#   - Sets:
#       litellm.callbacks.append("langfuse_otel")
#   Note: Uses "langfuse_otel" (OpenTelemetry-based) for Langfuse SDK v3+ compatibility

# 2. Create workflow with LangFuseTracer
tracer = LangFuseTracer(enable_runtime_graph=True)
wf = create_article_workflow(..., tracer=tracer)

# 3. Execute workflow
wf.execute(...)
# → LangFuseTracer creates spans and sets OpenTelemetry context
# → When tasks use context.llm_client, LLMClient is initialized with tracing
# → LiteLLM's Langfuse OTEL callback detects OTEL context automatically
# → LLM calls appear as nested spans in Langfuse
```

## Expected Langfuse Output

You should see traces like:

```
📊 Trace: article_article_0
  └─ 🔹 Span: search_article_0
  └─ 🔹 Span: curate_article_0
      └─ 🤖 Generation: litellm.completion (model: gpt-4o-mini)
  └─ 🔹 Span: write_article_0
      └─ 🤖 Generation: litellm.completion (model: gpt-4o-mini)
  └─ 🔹 Span: critique_article_0
      └─ 🤖 Generation: litellm.completion (model: gpt-4o-mini)
  └─ 🔹 Span: design_article_0
```

## Troubleshooting

### Issue: No traces appear in Langfuse

**Check 1: Verify credentials**
```bash
# Check environment variables are set
echo $LANGFUSE_PUBLIC_KEY
echo $LANGFUSE_SECRET_KEY
```

**Check 2: Verify OpenTelemetry is installed**
```bash
python -c "from opentelemetry import trace; print('OpenTelemetry available')"
```

**Check 3: Check logs**
Look for these messages in console output:
```
INFO:graflow.llm.client:Langfuse tracing enabled for LiteLLM (success + failure callbacks)
[article_0] LangFuse tracer initialized
```

Note: Tracing is automatically enabled when LLMClient is first initialized. If you see errors about missing credentials, ensure your `.env` file is properly configured.

### Issue: Workflow traces appear but no LLM calls

**Check 1: Verify LiteLLM callbacks are enabled**
```python
import litellm
print(litellm.callbacks)  # Should contain "langfuse_otel"
```

**Check 2: Verify OpenTelemetry context is set**
Add debug logging in your workflow:
```python
import logging
logging.getLogger("graflow.trace.langfuse").setLevel(logging.DEBUG)
```

You should see:
```
DEBUG:graflow.trace.langfuse:Set OpenTelemetry context for span 'write_article_0': trace_id=abcd1234..., span_id=ef567890...
```

**Check 3: Test LiteLLM Langfuse integration directly**
```python
import os
from litellm import completion

# Ensure env vars are set
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."

# Enable Langfuse OTEL callback (for SDK v3+)
import litellm
litellm.callbacks.append("langfuse_otel")

# Make a test call
response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}],
    metadata={"generation_name": "test"}
)

print("Check Langfuse UI for traces")
```

### Issue: Traces appear but not nested correctly

This usually means OpenTelemetry context is not being propagated. Verify:

1. **OpenTelemetry packages are installed:**
   ```bash
   pip list | grep opentelemetry
   ```
   Should show both `opentelemetry-api` and `opentelemetry-sdk`

2. **LangFuseTracer is being used:**
   ```python
   # In execute_article_workflow()
   tracer = LangFuseTracer(enable_runtime_graph=True)  # ✅ Correct
   # NOT None or NoopTracer
   ```

## Google ADK Integration (Optional)

**GoogleADKInstrumentor is NOT required for this example!**

It's only needed if you're using Google ADK's `LlmAgent` with ReAct/Supervisor patterns.

The GPT Newspaper example uses:
- ✅ Simple Python agent classes (WriterAgent, CuratorAgent, CritiqueAgent)
- ✅ LiteLLM via Graflow's LLMClient
- ❌ NOT using Google ADK's LlmAgent

### When to use Google ADK

Only if you're using Google ADK agents in your workflow:

```python
from google.adk.agents import LlmAgent
from graflow.llm.agents import AdkLLMAgent

# Create ADK agent with tools
adk_agent = LlmAgent(
    name="researcher",
    model="gemini-2.0-flash-exp",
    tools=[search_tool]
)

# Wrap for Graflow (instrumentation happens automatically)
agent = AdkLLMAgent(adk_agent, app_name=context.session_id)
# ✅ GoogleADKInstrumentor is called automatically (enable_tracing=True by default)

# To disable automatic tracing:
# agent = AdkLLMAgent(adk_agent, app_name=context.session_id, enable_tracing=False)
```

**Note**: `AdkLLMAgent` automatically calls `GoogleADKInstrumentor().instrument()` when initialized with `enable_tracing=True` (default). You don't need to manually instrument ADK!

## References

- [LiteLLM Langfuse Integration](https://docs.litellm.ai/docs/observability/langfuse_integration)
- [Langfuse Documentation](https://langfuse.com/docs)
- [Graflow LLM Integration Design](../../../docs/llm_integration_design.md)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)
