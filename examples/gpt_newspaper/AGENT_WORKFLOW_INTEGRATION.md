# Agent Workflow Integration Summary

## Overview

Successfully integrated the new **Agent Workflow** (`newspaper_agent_workflow.py`) into the GPT Newspaper application, making it accessible through both the backend API and frontend UI.

---

## Changes Made

### 1. Backend Configuration (`config.py`)

**Added:**
```python
# Agent Model Configuration
# GPT_NEWSPAPER_MODEL: Model for agent workflows (supports both Gemini and LiteLLM models)
# - For AdkLLMAgent: Use LiteLLM-compatible models (OpenAI, Anthropic, Gemini via LiteLLM, etc.)
# - For LiteLLM-based simple tasks: Any LiteLLM-compatible model
AGENT_MODEL = os.getenv("GPT_NEWSPAPER_MODEL", "gpt-4o-mini")
```

**Updated:**
- `Config.display()` to show agent model configuration

---

### 2. Agent Workflow (`newspaper_agent_workflow.py`)

**Added Model Configuration:**
```python
# Uses Config.AGENT_MODEL for agents (via GPT_NEWSPAPER_MODEL env var)
# Uses Config.DEFAULT_MODEL for simple LLM tasks (via GRAFLOW_LLM_MODEL env var)

# In agent registration:
agent_model = Config.AGENT_MODEL
research_agent = LlmAgent(name="researcher", model=agent_model, tools=[...])
editorial_agent = LlmAgent(name="editor", model=agent_model, tools=[...])

# In simple LLM tasks:
structured = llm.completion_text(messages, model=Config.DEFAULT_MODEL)
content = llm.completion_text(messages, model=Config.DEFAULT_MODEL)
```

**Documentation Updates:**
- Added comprehensive model configuration documentation in docstring
- Explained two-tier model configuration (GPT_NEWSPAPER_MODEL for agents, GRAFLOW_LLM_MODEL for LLM tasks)
- Added usage examples with different model combinations

---

### 3. Backend API (`api.py`)

**Updated Workflow Support:**
```python
# Added agent workflow import
from newspaper_agent_workflow import run_newspaper_workflow as run_agent_newspaper_workflow

# Updated workflow type
SupportedWorkflow = Literal["original", "dynamic", "agent"]

# Added agent workflow to runners
WORKFLOW_RUNNERS: Dict[SupportedWorkflow, Callable[..., str]] = {
    "original": run_original_newspaper_workflow,
    "dynamic": run_dynamic_newspaper_workflow,
    "agent": run_agent_newspaper_workflow,  # NEW
}

# Updated workflow field description
workflow: SupportedWorkflow = Field(
    default=DEFAULT_WORKFLOW,
    description="Workflow variant to execute: 'original' (simple LLM tasks), "
                "'dynamic' (parallel execution), or 'agent' (LLM agents with tools)",
    example=DEFAULT_WORKFLOW,
)
```

---

### 4. Frontend Types (`src/api/types.ts`)

**Updated:**
```typescript
export type WorkflowOption = "original" | "dynamic" | "agent";
```

---

### 5. Frontend API Client (`src/api/client.ts`)

**Updated:**
```typescript
export const supportedWorkflows: WorkflowOption[] = ["original", "dynamic", "agent"];
```

---

### 6. Frontend UI (`src/components/QueryForm.tsx`)

**Added Workflow Descriptions:**
```typescript
const workflowDescriptions: Record<WorkflowOption, string> = {
  original: "Simple LLM tasks with basic critique loop",
  dynamic: "Complex parallel tasks with quality gates and runtime task generation",
  agent: "LLM Agents with autonomous tool calling and ReAct pattern"
};
```

**Updated Workflow Selection UI:**
- Added agent workflow to toggle button group
- Updated description text to show all three workflows
- Added "View agent" link to preview agent workflow details

**Added Agent Workflow Preview Dialog:**
- Shows detailed architecture of research and editorial agents
- Lists tools for each agent
- Displays workflow flow diagram (text-based)
- Includes:
  - Research Agent: web search, fact extraction, query refinement (ReAct pattern)
  - Editorial Agent: fact checking, readability assessment, source verification, autonomous approval/revision decision

---

## Environment Variables

### Required (Agent Workflow)
```bash
TAVILY_API_KEY=tvly-...      # For web search tool
```

### Optional (Model Configuration)
```bash
# Agent model (for Research/Editorial agents)
GPT_NEWSPAPER_MODEL="gpt-4o-mini"  # Default
# Or use any LiteLLM-compatible model: "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp", etc.

# LLM model (for Curate/Write tasks)
GRAFLOW_LLM_MODEL="gpt-4o-mini"  # Default
# Or use: "gpt-4o", "claude-3-5-sonnet-20241022", etc.
```

---

## Model Configuration Strategy

The agent workflow uses **two separate model configurations**:

### 1. Agent Model (GPT_NEWSPAPER_MODEL)
- **Used by:** Research Agent, Editorial Agent (via Google ADK + LiteLLM bridge)
- **Supports:** Any LiteLLM-compatible chat model (OpenAI, Anthropic, Gemini, etc.)
- **Examples:** `"gpt-4o-mini"`, `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`, `"gemini-2.0-flash-exp"`
- **Default:** `"gpt-4o-mini"`

### 2. LLM Model (GRAFLOW_LLM_MODEL)
- **Used by:** Curate task, Write task (via LiteLLM)
- **Supports:** Any LiteLLM-compatible model
- **Examples:** `"gpt-4o"`, `"gpt-4o-mini"`, `"claude-3-5-sonnet-20241022"`
- **Default:** `"gpt-4o-mini"`

### Rationale
- **Cost Optimization**: Use cheaper models for simple generative tasks (writing)
- **Quality Optimization**: Use powerful models for agent reasoning (research, editorial with tools)
- **Flexibility**: Mix Gemini agents with OpenAI/Claude for writing

---

## Usage Examples

### Example 1: Default Configuration
```bash
export TAVILY_API_KEY="tvly-..."

# Runs with defaults:
# - Agents: gpt-4o-mini
# - LLM tasks: gpt-4o-mini
python newspaper_agent_workflow.py
```

### Example 2: High-Performance Agents
```bash
export GPT_NEWSPAPER_MODEL="gpt-4o"  # Better reasoning
export GRAFLOW_LLM_MODEL="gpt-4o"            # Better writing
python newspaper_agent_workflow.py
```

### Example 3: Cost-Optimized with Claude
```bash
export GPT_NEWSPAPER_MODEL="claude-3-5-sonnet-20241022"  # Agents via LiteLLM
export GRAFLOW_LLM_MODEL="claude-3-5-sonnet-20241022"  # Writing (any LiteLLM model)
python newspaper_agent_workflow.py
```

### Example 4: Via Frontend
1. Start backend: `uvicorn api:app --reload --port 8000`
2. Open frontend at `http://localhost:8000/frontend`
3. Select **"agent"** workflow from dropdown
4. Click "Generate newspaper"

---

## UI Features

### Workflow Selection
The frontend now displays three workflow options in a toggle button group:
- **original** - Simple workflow
- **dynamic** - Complex parallel workflow
- **agent** - Agent workflow (NEW)

### Workflow Descriptions
Inline descriptions show:
- **Original:** Simple LLM tasks with basic critique loop
- **Dynamic:** Complex parallel tasks with quality gates and runtime task generation
- **Agent:** LLM Agents with autonomous tool calling and ReAct pattern

### Workflow Preview
Clicking "View agent" opens a dialog showing:
- **Research Agent Architecture**
  - Autonomous web search with Tavily API
  - Extract key facts from multiple sources
  - Refine queries based on findings (ReAct pattern)

- **Editorial Agent Architecture**
  - Check factual claims against sources
  - Assess readability with textstat metrics
  - Verify source credibility
  - Autonomously decide: APPROVE or REVISE

- **Workflow Flow:** topic_intake → research_agent → curate → write → editorial_agent → design

---

## Testing

### Test Backend API
```bash
curl -X POST http://localhost:8000/api/newspaper \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["Latest AI developments"],
    "layout": "two-column",
    "workflow": "agent"
  }'
```

### Test Frontend
1. Set environment variables
2. Start backend: `cd examples/gpt_newspaper/backend && uvicorn api:app --reload`
3. Open: `http://localhost:8000/frontend`
4. Select "agent" workflow
5. Enter queries and click "Generate newspaper"

---

## Key Benefits

1. **Autonomous Decision Making**: Agents decide tool usage and revision loops
2. **Quality Verification**: Editorial agent performs systematic fact-checking
3. **Transparent Execution**: All tool calls visible in console output and traces
4. **Flexible Models**: Separate configuration for agent vs. LLM tasks
5. **Cost Optimization**: Use cheaper models where appropriate
6. **Seamless Integration**: Works alongside existing original and dynamic workflows

---

## File Summary

### Modified Files
- `backend/config.py` - Added AGENT_MODEL configuration
- `backend/newspaper_agent_workflow.py` - Updated to use configurable models
- `backend/api.py` - Added agent workflow support
- `frontend/src/api/types.ts` - Added "agent" to WorkflowOption
- `frontend/src/api/client.ts` - Added "agent" to supportedWorkflows
- `frontend/src/components/QueryForm.tsx` - Added agent workflow UI

### New Features
- ✅ Agent workflow selectable in UI
- ✅ Configurable models via environment variables
- ✅ Detailed agent preview dialog
- ✅ Two-tier model configuration (agents vs. LLM tasks)
- ✅ Comprehensive documentation

---

## Next Steps

### Optional Enhancements
1. **Add Agent Workflow Diagram**: Create visual SVG/PNG for frontend preview
2. **Model Validation**: Add frontend validation for agent model requirements
3. **Cost Estimation**: Show estimated cost per workflow in UI
4. **Agent Metrics**: Display agent tool call statistics in results
5. **Model Selection UI**: Add model dropdowns in frontend for advanced users

---

## Compatibility

- **Backend API:** Fully backward compatible with existing workflows
- **Frontend:** Seamlessly integrates with current UI
- **Environment Variables:** All new variables are optional with sensible defaults
- **Workflows:** All three workflows (original, dynamic, agent) coexist

---

## Documentation

See also:
- `WORKFLOW_COMPARISON.md` - Comparison of all three workflows
- `AGENT_WORKFLOW_DIAGRAM.md` - Visual architecture and examples
- `newspaper_agent_workflow.py` - Full implementation with docstrings
- `docs/llm_integration_design.md` - LLM integration design principles
