# GPT Newspaper - Graflow Workflow Examples

> **📊 LangFuse Tracing**: For setup and troubleshooting, see [LANGFUSE_SETUP.md](LANGFUSE_SETUP.md)

## Overview

This directory contains two versions of an **AI-powered newspaper article generation system**, designed to help you progressively understand Graflow's dynamic task generation capabilities.

## Two Workflow Versions

### 1. `newspaper_workflow.py` - **Basic Version**

A simple, easy-to-understand workflow perfect for learning Graflow's core features.

**Processing Flow:**
```
Search → Curation → Writing → Critique → Design
```

**Key Features:**
- ✅ Write-critique loop using `goto=True`
- ✅ Channel-based state management
- ✅ Parallel processing of multiple articles with ThreadPoolExecutor

### 2. `newspaper_dynamic_workflow.py` - **Advanced Version**

A production-ready advanced workflow leveraging all of Graflow's capabilities.

**Processing Flow:**
```
Topic Analysis → Multi-angle Search → Information Curation →
Multi-style Writing → Critique-Improvement Loop → Quality Checks → Design
```

**Key Features:**
- ✅ Runtime dynamic task expansion (`context.next_task()`)
- ✅ Dynamic gap filling (`context.next_iteration()`)
- ✅ Parallel writer personas (BestEffortGroupPolicy)
- ✅ Quality gates (AtLeastNGroupPolicy)

---

## How to Run

```bash
# Set environment variables
export TAVILY_API_KEY="your_tavily_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Run basic version
PYTHONPATH=. uv run python examples/gpt_newspaper/backend/newspaper_workflow.py

# Run advanced version
PYTHONPATH=. uv run python examples/gpt_newspaper/backend/newspaper_dynamic_workflow.py
```

---

## Learning Path

**Recommended Learning Order:**

1. **Start with `newspaper_workflow.py`** - Understand basic patterns
2. **Progress to `newspaper_dynamic_workflow.py`** - Learn advanced features

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

## Workflow Comparison

| Feature | Basic (`newspaper_workflow.py`) | Advanced (`newspaper_dynamic_workflow.py`) |
|---------|--------------------------------|-------------------------------------------|
| **Task Count** | 5 tasks | 15+ tasks (dynamically grows) |
| **Search Strategy** | Single query | Multi-angle search based on topic analysis |
| **Dynamic Task Generation** | None (goto=True only) | Search expansion, gap filling, supplemental research |
| **Writing Approach** | Single style | 3 personas in parallel → select best |
| **Quality Assurance** | Critique loop only | Critique + 3-stage quality gate |
| **Parallel Execution** | Article level (ThreadPoolExecutor) | Task level + article level |
| **Error Handling** | Iteration limit only | Partial success tolerance (AtLeastNGroupPolicy) |
| **Lines of Code** | ~325 lines | ~604 lines |
| **Learning Curve** | 🟢 Beginner-friendly | 🟡 Intermediate to Advanced |

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

These two workflows are designed to progressively understand **Graflow's dynamic task generation capabilities**.

### What You Learn from the Basic Version

- ✅ Loop-back to existing tasks with `goto=True`
- ✅ State management with channels
- ✅ Iteration control (prevent infinite loops)
- ✅ Parallel execution of multiple workflows

### What You Learn from the Advanced Version

- ✅ Dynamic task generation at runtime (`context.next_task()`)
- ✅ Self-iteration for waiting/retry (`context.next_iteration()`)
- ✅ Parallel groups and execution policies (BestEffortGroupPolicy, AtLeastNGroupPolicy)
- ✅ Complex state management and task coordination

### Comparison with Traditional Workflow Engines

| Feature | Traditional Workflow | Graflow (Basic) | Graflow (Advanced) |
|---------|---------------------|-----------------|-------------------|
| Task Definition | All pre-defined | Pre-defined + goto | Runtime dynamic generation |
| Cyclic Flows | Not supported (DAG only) | Loop-back with `goto=True` | Multiple loop patterns |
| Conditional Branching | Pre-definition required | Runtime state-based | Dynamic task addition for branching |
| Parallel Execution | Fixed parallel tasks | Workflow level | Task level + workflow level |
| Error Handling | Fixed retry | Iteration limits | Dynamic gap-filling task addition + partial success tolerance |

### Real-World Use Cases

These design patterns can be applied to real-world use cases such as:

- 📰 **Content Generation Pipelines**: Iterative generation with quality checks
- 🤖 **Multi-Agent Systems**: Agent collaboration with feedback loops
- 🔄 **Iterative Improvement Workflows**: Automatic improvement until quality standards are met
- 📊 **Data Analysis Pipelines**: Dynamic processing expansion based on data quality
- ⚙️ **ETL Pipelines**: Automatic gap-filling when data sources are insufficient

---

## References

- [Graflow Documentation](https://github.com/myui/graflow)
- [Dynamic Tasks Example](../../07_dynamic_tasks/)
- [Parallel Execution Example](../../08_workflow_composition/)
