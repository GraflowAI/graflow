# GPT Newspaper - Graflow Workflow Examples

> **📊 LangFuse Tracing**: For setup and troubleshooting, see [LANGFUSE_SETUP.md](LANGFUSE_SETUP.md)

## Overview

This directory contains **three versions** of an **AI-powered newspaper article generation system**, each demonstrating different Graflow patterns and use cases.

## Workflow Comparison

| Feature | Simple | Dynamic | **Agent** |
|---------|--------|---------|-----------|
| **File** | `newspaper_workflow.py` | `newspaper_dynamic_workflow.py` | `newspaper_agent_workflow.py` |
| **Task Count** | 5 tasks | 15+ tasks (dynamic) | 6 tasks |
| **Complexity** | 🟢 Low | 🟡 High | 🟡 Medium |
| **LLM Agents** | 0 (pure LLM tasks) | 0 (pure LLM tasks) | **2 (researcher, editor)** |
| **Tool Calling** | ❌ No | ❌ No | **✅ Yes (8 tools)** |
| **Research** | Single search | Multi-angle fan-out | **Autonomous multi-search** |
| **Writing** | Single LLM call | 3 parallel personas | Single LLM with revision |
| **Quality Control** | Simple critique | Critique + 3 gates | **Agent-driven editorial** |
| **Revision Loop** | Fixed logic (goto) | Fixed logic (goto) | **Agent-controlled** |
| **Dynamic Tasks** | None (goto only) | Search expansion, gap filling | None (agents decide) |
| **Parallel Execution** | Article level | Task + article level | Article level |
| **Error Handling** | Iteration limit | Partial success (2/3) | Agent judgment |
| **Latency** | ~30s | ~45s | ~60s |
| **LLM Calls** | 4-6 | 10-15 | 8-12 + tools |
| **Cost (relative)** | 1x | 2-3x | 1.5-2x |
| **Quality** | Good | Very Good | **Excellent** |
| **Autonomy** | Low | Medium | **High** |
| **Transparency** | Medium | Medium | **High (tools visible)** |
| **Lines of Code** | ~325 | ~604 | ~1100 |
| **Learning Curve** | 🟢 Beginner | 🟡 Intermediate | 🟡 Intermediate |

## Decision Guide

### Choose Simple Workflow if:
- ✅ Learning Graflow basics
- ✅ Cost is primary concern
- ✅ Simple content generation
- ✅ No complex requirements

### Choose Dynamic Workflow if:
- ✅ Need parallel execution
- ✅ Multiple processing approaches
- ✅ Quality gates and validation
- ✅ Production scale

### Choose Agent Workflow if:
- ✅ Quality is critical (fact-checking/verification)
- ✅ Need autonomous decision-making
- ✅ Want transparent tool usage (visible in traces)
- ✅ Building agentic systems
- ✅ Exploring LLM agent patterns

---

## How to Run

### Simple Workflow
```bash
# Set environment variables
export TAVILY_API_KEY="your_tavily_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Run
PYTHONPATH=. uv run python examples/gpt_newspaper/backend/newspaper_workflow.py
```

### Dynamic Workflow
```bash
# Same environment variables as simple
export TAVILY_API_KEY="your_tavily_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Run
PYTHONPATH=. uv run python examples/gpt_newspaper/backend/newspaper_dynamic_workflow.py
```

### Agent Workflow
```bash
# Install additional dependencies
uv add google-adk textstat tavily-python

# Set environment variables
export TAVILY_API_KEY="your_tavily_api_key"
export GPT_NEWSPAPER_MODEL="gpt-4o-mini"  # Agent model (optional)
export GRAFLOW_LLM_MODEL="gpt-4o-mini"    # LLM tasks (optional)
export OPENAI_API_KEY="your_openai_api_key"  # Or other provider

# Run
PYTHONPATH=. uv run python examples/gpt_newspaper/backend/newspaper_agent_workflow.py
```

**For detailed agent workflow setup:** See [agent_workflow.md](agent_workflow.md)

---

## Learning Path

**Recommended Order:** Simple → Dynamic → Agent

1. **`newspaper_workflow.py`** - Learn goto loops, channels, state management
2. **`newspaper_dynamic_workflow.py`** - Learn dynamic task generation, parallel execution, group policies
3. **`newspaper_agent_workflow.py`** - Learn LLM agents, tool calling, autonomous decision-making

---

# Part 1: Basic Version (`newspaper_workflow.py`)

## Workflow Overview

```
search → curate → write ⟲ critique → design
                    ↑      ↓
                    └──────┘ (feedback loop)
```

## Core Feature: Loop-back with goto=True

`newspaper_workflow.py:123-170`

```python
@task(id=f"critique_{article_id}", inject_context=True)
def critique_task(context: TaskExecutionContext) -> Dict:
    """Critique the article and loop back to writer if feedback exists"""
    channel = context.get_channel()
    article = channel.get("article")
    iteration = channel.get("iteration", default=0)

    result = critique_agent.run(article)
    channel.set("article", result)

    if result.get("critique") is not None:
        # Has feedback
        if iteration >= 5:
            print(f"⚠️ Max iterations reached")
            result["critique"] = None  # Force approval
        else:
            channel.set("iteration", iteration + 1)
            # ★ Loop back to existing write_task
            context.next_task(write_task, goto=True)
            return result

    # Approved - naturally proceed to design_task
    return result
```

### State Management with Channels

```python
# Save state
channel.set("article", result)
channel.set("iteration", iteration + 1)

# Retrieve state
article = channel.get("article")
iteration = channel.get("iteration", default=0)
```

---

# Part 2: Advanced Version (`newspaper_dynamic_workflow.py`)

## Workflow Overview

```
topic_intake
    ↓
search_router (dynamically generates multiple search tasks)
    ↓ ↓ ↓ ↓
  search_angle_1 | search_angle_2 | search_angle_3 | ...
    ↓ ↓ ↓ ↓
curate ⟲ (adds supplemental research if insufficient)
    ↓
writer_personas (parallel execution)
    ↓ ↓ ↓
  write_feature | write_brief | write_data_digest
    ↓ ↓ ↓
select_draft (choose best draft)
    ↓
write ⟲←──┐
    ↓      │
critique ──┘ (loop back if feedback exists)
    ↓
quality_gate (parallel execution, pass with 2/3 success)
    ↓ ↓ ↓
  fact_check | compliance_check | risk_check
    ↓ ↓ ↓
quality_gate_summary
    ↓
design
```

## Three Key Advanced Features of Graflow

### 1. **Dynamic Search Task Expansion** 📡

`newspaper_dynamic_workflow.py:149-175`

```python
@task(id=f"search_router_{article_id}", inject_context=True)
def search_router_task(context: TaskExecutionContext) -> Dict:
    """Dynamically determine multiple search angles based on topic"""
    channel = context.get_channel()
    angles: List[str] = channel.get("angles", default=["overview"])

    print(f"[{article_id}] 🔍 Launching {len(angles)} targeted searches...")

    for angle in angles:
        angle_id = _slugify(angle)

        def run_angle(angle_label=angle, angle_slug=angle_id):
            task_query = f"{query} - focus on {angle_label}"
            result = search_agent.run({"query": task_query, "angle": angle_label})
            # Aggregate results in channel
            aggregated = channel.get("search_results", default=[])
            aggregated.append(result)
            channel.set("search_results", aggregated)
            return result

        # ★ Dynamically add task at runtime
        angle_task = TaskWrapper(f"search_{article_id}_{angle_id}", run_angle)
        context.next_task(angle_task)

    return {"scheduled": len(angles)}
```

#### What It Does

- Determines multiple search angles (policy, market outlook, climate impact, technology) based on topic analysis
- **Dynamically generates search tasks at runtime** for each angle (`context.next_task()`)
- Executes multiple searches in parallel and aggregates results in channel

#### Difference from Traditional Workflow Engines

- ✅ No need to know the number of search tasks in advance
- ✅ Flexibly expand search scope based on topic content
- ✅ Workflow self-adapts based on runtime state

---

### 2. **Dynamic Gap Filling** 🔄

`newspaper_dynamic_workflow.py:205-229`

```python
@task(id=f"curate_{article_id}", inject_context=True)
def curate_task(context: TaskExecutionContext) -> Dict:
    """Add supplemental research if sources are insufficient during curation"""
    channel = context.get_channel()
    expected = channel.get("expected_search_tasks", default=0)
    completed = channel.get("completed_search_tasks", default=0)

    # Wait for search tasks to complete
    if expected and completed < expected:
        print(f"[{article_id}] ⏳ Waiting for {expected - completed} search tasks...")
        time.sleep(0.05)
        context.next_iteration()  # ★ Re-execute self
        return {"status": "waiting"}

    # Curate sources
    result = curator_agent.run(article_data)

    # If sources are insufficient
    min_sources = 3
    if len(result.get("sources", [])) < min_sources and not channel.get("gap_fill_requested"):
        channel.set("gap_fill_requested", True)

        def supplemental_research():
            # Execute supplemental research
            supplemental_result = search_agent.run({
                "query": f"{query} statistics and data",
                "angle": "data supplement"
            })
            # Aggregate results
            aggregated = channel.get("search_results", default=[])
            aggregated.append(supplemental_result)
            channel.set("search_results", aggregated)
            return supplemental_result

        # ★ Dynamically add supplemental research task
        context.next_task(TaskWrapper(f"search_{article_id}_supplemental", supplemental_research))
        print(f"[{article_id}] 🔄 Not enough sources, scheduling supplemental research...")
        context.next_iteration()  # Execute curation again
        return {"status": "gap_filling"}

    return result
```

#### What It Does

- Detects insufficient information sources during curation (minimum 3 sources required)
- **Dynamically adds** supplemental research task (`context.next_task()`)
- **Re-executes** curation task itself (`context.next_iteration()`) to wait for supplemental information

#### Key Points

- ✅ Automatically expands information gathering until quality standards are met
- ✅ Flexible response based on runtime state
- ✅ `gap_fill_requested` flag prevents infinite loops

---

### 3. **Write-Critique Iteration Loop** 🔁

`newspaper_dynamic_workflow.py:329-359`

```python
@task(id=f"critique_{article_id}", inject_context=True)
def critique_task(context: TaskExecutionContext) -> Dict:
    """Critique article and loop back to write task if feedback exists"""
    print(f"[{article_id}] 🔎 Critiquing article...")

    channel = context.get_channel()
    article = _get_article_from_channel(context)
    iteration = channel.get("iteration", default=0)

    # Critique agent reviews article
    result = critique_agent.run(article)
    channel.set("article", result)

    # If feedback exists
    if result.get("critique") is not None:
        print(f"[{article_id}] 🔄 Critique feedback received, looping back to writer...")

        if iteration >= MAX_REVISION_ITERATIONS:
            print(f"[{article_id}] ⚠️ Max iterations ({MAX_REVISION_ITERATIONS}) reached")
            result["critique"] = None  # Force approval
        else:
            channel.set("iteration", iteration + 1)
            # ★ Loop back to existing write_task
            context.next_task(write_task, goto=True)
            return result

    print(f"[{article_id}] ✅ Article approved by critique!")
    return result
```

#### What It Does

1. Critique agent reviews the article
2. If feedback exists, **loop back to existing write_task** (`goto=True`)
3. Repeat write-critique cycle up to 3 times
4. If critique approves, proceed to quality check phase

#### Difference from Traditional Workflows

- ✅ Enables cyclic flows not expressible in static DAGs (Directed Acyclic Graphs)
- ✅ Dynamically iterate until quality is met
- ✅ Iteration limit prevents infinite loops (`MAX_REVISION_ITERATIONS = 3`)

---

## Other Graflow Features

### 4. **Parallel Writer Personas** ✍️

`newspaper_dynamic_workflow.py:263-268`

```python
writer_personas = (
    write_feature_task | write_brief_task | write_data_digest_task
).with_execution(
    backend=CoordinationBackend.THREADING,
    policy=BestEffortGroupPolicy(),
)
```

- Execute 3 writing styles (feature, brief, data_digest) **in parallel**
- `BestEffortGroupPolicy()`: Best-effort execution, select the best output
- Thread-based parallel execution for efficiency

### 5. **Quality Gate with Partial Success Tolerance** 🛡️

`newspaper_dynamic_workflow.py:385-390`

```python
quality_gate = (
    fact_check_task | compliance_check_task | risk_check_task
).with_execution(
    backend=CoordinationBackend.THREADING,
    policy=AtLeastNGroupPolicy(min_success=2),  # ★ Minimum 2 successes required
)
```

- Execute fact check, compliance, and risk assessment in parallel
- **Pass with minimum 2 successes** (tolerates partial failures)
- Enables flexible quality management in production environments

---

## Key Insights

### Simple → Dynamic
**Progression**: Learn parallel execution, runtime task generation, group policies

### Simple → Agent
**Progression**: Learn LLM agents, tool calling, autonomous decision-making

### Dynamic → Agent
**Comparison**: Dynamic has more parallel tasks, Agent has autonomous reasoning

### All Three
**Study**: See evolution from fixed logic → parallel complexity → autonomous agents

---

## State Management with Channels

Both workflows leverage **channels** for sharing state across tasks:

```python
channel = context.get_channel()

# Aggregate search results
channel.set("search_results", aggregated)
channel.get("search_results", default=[])

# Iteration management
channel.set("iteration", iteration + 1)
channel.get("iteration", default=0)

# Share article data
channel.set("article", result)
channel.get("article")

# Flag management
channel.set("gap_fill_requested", True)
channel.get("gap_fill_requested", default=False)
```

### Channel Benefits

- ✅ Share data across tasks
- ✅ Maintain state across iterations
- ✅ Dynamic tasks can access the same channel

---

## Summary

These workflows demonstrate **Graflow's evolution from simple loops to autonomous agents**, each building on the previous pattern.

### Graflow vs Traditional Workflow Engines

| Feature | Traditional | Graflow (Simple) | Graflow (Dynamic) | Graflow (Agent) |
|---------|------------|-----------------|-------------------|-----------------|
| **Task Definition** | All pre-defined | Pre-defined + goto | Runtime dynamic generation | Pre-defined + agents |
| **Cyclic Flows** | Not supported (DAG only) | Loop-back with `goto=True` | Multiple loop patterns | Agent-controlled loops |
| **Conditional Branching** | Pre-definition required | Runtime state-based | Dynamic task addition | Agent decides |
| **Parallel Execution** | Fixed parallel tasks | Workflow level | Task + workflow level | Workflow level |
| **Error Handling** | Fixed retry | Iteration limits | Dynamic gap-filling + partial success | Agent judgment |
| **Autonomy** | None | Low | Medium | **High** |

### Real-World Use Cases

**Simple Workflow:**
- 📰 Basic content generation with review cycle
- 📝 Document processing with quality checks

**Dynamic Workflow:**
- 🔄 Complex content pipelines with multi-angle research
- 📊 Data analysis with dynamic processing expansion
- ⚙️ ETL pipelines with automatic gap-filling

**Agent Workflow:**
- 🔍 Research automation with autonomous verification
- 🤖 Agentic systems with self-directed tool usage
- ✅ Quality-critical tasks (fact-checking, compliance)
- 🧠 Decision support with transparent reasoning

---

## References

- [Graflow Documentation](https://github.com/myui/graflow)
- [Dynamic Tasks Example](../../07_dynamic_tasks/)
- [Parallel Execution Example](../../08_workflow_composition/)
