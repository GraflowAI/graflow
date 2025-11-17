# Agent Workflow Setup Guide

## Prerequisites

The agent workflow requires additional dependencies beyond the base newspaper workflows.

### Required Dependencies

```bash
# Navigate to backend directory
cd examples/gpt_newspaper/backend

# Install agent workflow dependencies
uv add google-adk textstat tavily-python

# Or with pip
pip install google-adk textstat tavily-python
```

### Required API Keys

```bash
# Tavily API (for web search tool)
export TAVILY_API_KEY="tvly-..."
# Get your key at: https://tavily.com/

# LLM provider API key(s) for your selected GPT_NEWSPAPER_MODEL / GRAFLOW_LLM_MODEL
# Example for OpenAI models:
# export OPENAI_API_KEY="sk-..."
# Example for Anthropic models:
# export ANTHROPIC_API_KEY="sk-ant-..."
# Example for Gemini via LiteLLM:
# export GOOGLE_API_KEY="..."
# (Any LiteLLM-supported provider works—configure the matching key before running.)
```

### Optional Configuration

```bash
# Agent model (default: "gpt-4o-mini")
export GPT_NEWSPAPER_MODEL="gpt-4o"

# LLM model for simple tasks (default: "gpt-4o-mini")
export GRAFLOW_LLM_MODEL="gpt-4o"

# Important: Configure the corresponding provider API key (OpenAI, Anthropic, Gemini, etc.)
# so LiteLLM can call your chosen models.
```

---

## Quick Start

### 1. Install Dependencies

```bash
cd examples/gpt_newspaper/backend

# Install with uv (recommended)
uv add google-adk textstat tavily-python

# Verify installation
python3 -c "import google.adk; import textstat; import tavily; print('✅ All dependencies installed')"
```

### 2. Set Environment Variables

Create a `.env` file:

```bash
# examples/gpt_newspaper/backend/.env
TAVILY_API_KEY=tvly-your-key-here
GPT_NEWSPAPER_MODEL=gpt-4o-mini
GRAFLOW_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-openai-key
```

Or export them:

```bash
export TAVILY_API_KEY="tvly-..."
export GPT_NEWSPAPER_MODEL="gpt-4o-mini"
# export OPENAI_API_KEY="sk-..."  # Replace with provider key for your models

echo "Remember to set whichever provider key matches GPT_NEWSPAPER_MODEL / GRAFLOW_LLM_MODEL."
# export OPENAI_API_KEY="sk-..."  # Or whichever provider your models require
```

### 3. Run the Workflow

#### Option A: Command Line

```bash
cd examples/gpt_newspaper/backend
PYTHONPATH=../../.. python newspaper_agent_workflow.py
```

#### Option B: Via API

```bash
# Start the backend
cd examples/gpt_newspaper/backend
uvicorn api:app --reload --port 8000

# In another terminal, test with curl
curl -X POST http://localhost:8000/api/newspaper \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Latest AI developments"],
    "layout": "two-column",
    "workflow": "agent"
  }'
```

#### Option C: Via Frontend

```bash
# Start backend (if not already running)
cd examples/gpt_newspaper/backend
uvicorn api:app --reload --port 8000

# Open frontend
# http://localhost:8000/frontend

# Select "agent" from workflow dropdown
# Click "Generate newspaper"
```

---

## Troubleshooting

### Error: "Google ADK not installed"

**Solution:**
```bash
uv add google-adk
# or
pip install google-adk
```

### Error: "TAVILY_API_KEY environment variable is required"

**Solution:**
1. Sign up at https://tavily.com/
2. Get your API key
3. Export it: `export TAVILY_API_KEY="tvly-..."`

### Warning: "Tavily not installed (web search disabled)"

**Impact:** Research agent cannot use web_search tool

**Solution:**
```bash
uv add tavily-python
# or
pip install tavily-python
```

### Warning: "textstat not installed (readability assessment disabled)"

**Impact:** Editorial agent cannot assess readability

**Solution:**
```bash
uv add textstat
# or
pip install textstat
```

---

## Verification Checklist

Before running the agent workflow, verify:

- [ ] Google ADK installed (`python -c "import google.adk"`)
- [ ] Tavily installed (`python -c "import tavily"`)
- [ ] textstat installed (`python -c "import textstat"`)
- [ ] TAVILY_API_KEY set (`echo $TAVILY_API_KEY`)
- [ ] GPT_NEWSPAPER_MODEL set (optional override, `echo $GPT_NEWSPAPER_MODEL`)
- [ ] Provider API key(s) for LiteLLM models (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)

Run all checks:

```bash
# Verify all dependencies
python3 << 'EOF'
import os
import sys

checks = {
    "Google ADK": lambda: __import__("google.adk"),
    "Tavily": lambda: __import__("tavily"),
    "textstat": lambda: __import__("textstat"),
    "TAVILY_API_KEY": lambda: os.getenv("TAVILY_API_KEY"),
    "GPT_NEWSPAPER_MODEL": lambda: os.getenv("GPT_NEWSPAPER_MODEL"),
    "LLM Provider Key": lambda: os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("GOOGLE_API_KEY"),
}

all_ok = True
for name, check in checks.items():
    try:
        result = check()
        if result:
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - Not set")
            all_ok = False
    except Exception as e:
        print(f"❌ {name} - {e}")
        all_ok = False

if all_ok:
    print("\n🎉 All requirements satisfied!")
    sys.exit(0)
else:
    print("\n⚠️  Some requirements missing. See above.")
    sys.exit(1)
EOF
```

---

## Model Configuration

The agent workflow uses **two separate models**:

### Agent Model (GPT_NEWSPAPER_MODEL)

**Used by:** Research Agent, Editorial Agent

**Requirements:**
- Must be any LiteLLM-compatible chat model
- Examples: `"gpt-4o-mini"`, `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`, `"gemini-2.0-flash-exp"`

**Configuration:**
```bash
export GPT_NEWSPAPER_MODEL="gpt-4o-mini"  # Default
# or
export GPT_NEWSPAPER_MODEL="gpt-4o"  # More powerful
```

### LLM Model (GRAFLOW_LLM_MODEL)

**Used by:** Curate task, Write task

**Requirements:**
- Any LiteLLM-compatible model
- Examples: `"gpt-4o"`, `"gpt-4o-mini"`, `"claude-3-5-sonnet-20241022"`

**Configuration:**
```bash
export GRAFLOW_LLM_MODEL="gpt-4o-mini"  # Default (cost-effective)
# or
export GRAFLOW_LLM_MODEL="gpt-4o"  # Higher quality
# or
export GRAFLOW_LLM_MODEL="claude-3-5-sonnet-20241022"  # Claude for writing
```

**Example Configurations:**

```bash
# Cost-optimized
export GPT_NEWSPAPER_MODEL="gpt-4o-mini"
export GRAFLOW_LLM_MODEL="gpt-4o-mini"

# Quality-optimized
export GPT_NEWSPAPER_MODEL="gpt-4o"
export GRAFLOW_LLM_MODEL="gpt-4o"

# Hybrid (Claude agents + Claude writing)
export GPT_NEWSPAPER_MODEL="claude-3-5-sonnet-20241022"
export GRAFLOW_LLM_MODEL="claude-3-5-sonnet-20241022"
```

---

## Cost Estimation

### Agent Workflow (per article)

**Agent Models (LiteLLM):**
- gpt-4o-mini: ~$0.01 per article
- gpt-4o: ~$0.05 per article
- claude-3-5-sonnet: ~$0.03 per article
- gemini-2.0-flash-exp: ~$0.01 per article

**LLM Models (writing):**
- gpt-4o-mini: ~$0.01 per article
- gpt-4o: ~$0.05 per article
- claude-3-5-sonnet: ~$0.03 per article

**Tavily Search:**
- ~$0.002 per search (typically 3-5 searches per article)

**Estimated Total:**
- Budget: ~$0.02-0.03 per article (flash + mini)
- Premium: ~$0.10-0.15 per article (pro + gpt-4o)

---

## Next Steps

Once setup is complete:

1. Test with a single query:
   ```bash
   PYTHONPATH=../../.. python newspaper_agent_workflow.py
   ```

2. Monitor agent output:
   - Tool calls will be visible in console
   - Agent reasoning displayed in real-time
   - Editorial decisions shown with justification

3. Integrate with frontend:
   - Start backend: `uvicorn api:app --reload`
   - Open: http://localhost:8000/frontend
   - Select "agent" workflow

---

## Additional Resources

- **Workflow Comparison:** `WORKFLOW_COMPARISON.md`
- **Architecture Diagram:** `AGENT_WORKFLOW_DIAGRAM.md`
- **Bug Fixes:** `BUGFIXES.md`
- **LLM Integration:** `../../docs/llm_integration_design.md`
- **Google ADK Docs:** https://developers.google.com/adk
- **Tavily API:** https://tavily.com/docs
