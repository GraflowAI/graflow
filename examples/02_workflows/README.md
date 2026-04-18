# 02 - Workflow Orchestration

**Difficulty**: Intermediate
**Status**: ✅ Complete
**Prerequisites**: Complete [01_basics](../01_basics/) first

## Overview

This section demonstrates **workflow orchestration** in Graflow - how to compose multiple tasks into coordinated workflows with sequential and parallel execution patterns.

## What You'll Learn

- 🔄 Using the `workflow()` context manager
- ➡️ Sequential execution with the `>>` operator
- ⚡ Parallel execution with the `|` operator
- 🎯 Accessing execution context with `inject_context`
- 📊 Building complex DAG (Directed Acyclic Graph) workflows

## Examples

### 1. simple_pipeline.py ⭐ START HERE

**Concept**: The simplest workflow - a basic 3-task pipeline

The "Hello World" of Graflow workflows. This is the absolute simplest example showing workflow basics.

```bash
uv run python examples/02_workflows/simple_pipeline.py
```

**Key Concepts**:
- Using the `workflow()` context manager
- Defining tasks with `@task`
- Sequential execution with `>>`
- Starting workflow execution

**Expected Output**:
```
=== Simple Pipeline Demo ===
...
Starting!
Middle!
End!
✅ Pipeline completed successfully!
```

---

### 2. workflow_decorator.py

**Concept**: Workflow context manager

Learn how to use the `workflow()` context manager to define and execute coordinated task workflows.

```bash
uv run python examples/02_workflows/workflow_decorator.py
```

**Key Concepts**:
- Creating workflow contexts
- Defining tasks within workflows
- Executing workflows with `ctx.execute()`
- Understanding workflow execution flow

---

### 3. operators_demo.py

**Concept**: Task composition operators

Master the `>>` (sequential) and `|` (parallel) operators for building complex workflows.

```bash
uv run python examples/02_workflows/operators_demo.py
```

**Key Concepts**:
- Sequential execution: `task1 >> task2`
- Parallel execution: `task1 | task2`
- Combined patterns: `(task1 | task2) >> task3`
- DAG construction

---

### 4. context_injection.py

**Concept**: Accessing execution context

Learn how to access the execution context within tasks for advanced control and state management.

```bash
uv run python examples/02_workflows/context_injection.py
```

**Key Concepts**:
- Using `inject_context=True`
- Accessing session information
- Using channels for inter-task communication
- Storing and retrieving task results

---

### 5. agent_loop.py

**Concept**: Cyclic agent ↔ tool workflow (three loop styles)

Demonstrates three ways to build iterative loops in Graflow, from static cycles to dynamic jumps to self-iteration.

```bash
uv run python examples/02_workflows/agent_loop.py
```

**Key Concepts**:
- **`agent_tool_loop()`** — Static cycle with `agent >> tool >> agent` + `terminate_workflow()`
- **`loop_with_goto()`** — Dynamic jumps with `next_task(t, goto=True)` + `@task(max_cycles=N)`
- **`loop_with_iteration()`** — Single-task self-loop with `next_iteration(data)` + `@task(max_cycles=N)`
- Cycle budget APIs: `ctx.cycle_count`, `ctx.max_cycles`, `ctx.can_iterate()`

**Expected Output**:
```
=== Agent Loop Demo ===
[iter 1] agent: thinking...
[iter 1] tool:  running tool (step=1)
[iter 2] agent: thinking...
[iter 2] tool:  running tool (step=2)
[iter 3] agent: thinking...
[iter 3] agent: condition met → terminate
Loop finished. Agent calls: 3, Tool calls: 2

=== Dynamic Jump Loop Demo ===
[cycle 1/5] agent: reflecting... (score=0.2)
           tool:  executing action (call #1)
...
[cycle 4/5] agent: quality threshold reached — done
Dynamic jump loop finished after 4 cycles. Final score: 0.8

=== Self-Refinement Loop Demo ===
[iter 1/5] refining... (draft_v1, score=0.3)
...
[iter 4/5] quality threshold reached — accepting draft_v4
[publish] publishing draft_v4
Self-refinement finished after 4 iterations. Published: draft_v4
```

---

### 6. task_graph_lowlevel_api.py

**Concept**: Low-level TaskGraph API usage

Learn how to construct workflows using the low-level TaskGraph APIs directly, without the high-level `workflow()` context or operators (`>>`, `|`).

```bash
uv run python examples/02_workflows/task_graph_lowlevel_api.py
```

**Key Concepts**:
- Direct TaskGraph manipulation with `add_node()` and `add_edge()`
- Creating ExecutionContext manually with `ExecutionContext.create()`
- Using `WorkflowEngine` for execution
- Understanding the relationship between high-level and low-level APIs
- When to use low-level APIs (dynamic workflows, custom tools, etc.)

---

## Workflow Orchestration Patterns

### Pattern 1: Simple Sequential Pipeline

```python
from graflow.core.decorators import task
from graflow.core.workflow import workflow

with workflow("pipeline") as ctx:
    @task
    def step1():
        print("Step 1")

    @task
    def step2():
        print("Step 2")

    step1 >> step2
    ctx.execute("step1")
```

### Pattern 2: Parallel Processing

```python
with workflow("parallel") as ctx:
    @task
    def task_a():
        print("Task A")

    @task
    def task_b():
        print("Task B")

    @task
    def combine():
        print("Combining results")

    # Both task_a and task_b run in parallel
    # Then combine runs after both complete
    (task_a | task_b) >> combine
    ctx.execute()
```

### Pattern 3: Complex DAG

```python
with workflow("dag") as ctx:
    @task
    def extract():
        return "data"

    @task
    def transform_a(data):
        return f"{data}_a"

    @task
    def transform_b(data):
        return f"{data}_b"

    @task
    def load(results):
        print(f"Loading: {results}")

    # Diamond pattern:
    # extract -> (transform_a | transform_b) -> load
    extract >> (transform_a | transform_b) >> load
    ctx.execute("extract")
```

## Execution Model

Graflow workflows execute tasks based on the dependency graph you define:

1. **Graph Construction**: Operators (`>>`, `|`) build a DAG of task dependencies
2. **Queue Management**: Tasks are queued when their dependencies complete
3. **Execution**: The workflow engine executes tasks in topological order
4. **Parallelism**: Tasks with no dependencies between them can run in parallel (with `|`)

## When to Use Workflows

Use workflow orchestration when you need:

- ✅ **Coordinated execution** of multiple tasks
- ✅ **Sequential pipelines** (ETL, data processing)
- ✅ **Parallel processing** of independent tasks
- ✅ **Complex dependencies** between tasks
- ✅ **Execution control** (max steps, start nodes)

Use direct task calls when you have:

- ❌ Simple, single-task execution
- ❌ No dependencies between tasks
- ❌ Dynamic task selection at runtime

## Common Patterns

### ETL Pipeline

```python
with workflow("etl") as ctx:
    extract >> transform >> load
    ctx.execute("extract")
```

### Fan-out / Fan-in

```python
with workflow("fan") as ctx:
    # Fan-out: one task triggers multiple parallel tasks
    fetch >> (process_a | process_b | process_c)

    # Fan-in: multiple tasks converge to one
    (task1 | task2 | task3) >> aggregate
```

### Multi-stage Processing

```python
with workflow("stages") as ctx:
    # Stage 1: Parallel data loading
    (load_db | load_api | load_file) >> validate

    # Stage 2: Sequential processing
    validate >> transform >> enrich

    # Stage 3: Parallel outputs
    enrich >> (export_json | export_csv | export_db)
```

## Debugging Tips

1. **Print execution flow**: Add print statements to see task execution order
2. **Use max_steps**: Limit execution steps during development
   ```python
   ctx.execute("start", max_steps=5)
   ```
3. **Check the graph**: Use `ctx.graph.nodes` to inspect registered tasks
4. **Monitor channels**: Use `inject_context=True` to inspect channel state

## Next Steps

After mastering workflow orchestration:

1. **03_data_flow**: Learn advanced inter-task communication
2. **04_execution**: Explore custom execution handlers
3. **05_distributed**: Scale workflows across multiple workers

## API Reference

**Workflow Context**:
- `workflow(name)` - Create a workflow context
- `ctx.execute(start_task, max_steps)` - Execute the workflow

**Operators**:
- `task1 >> task2` - Sequential: task2 runs after task1
- `task1 | task2` - Parallel: both tasks can run concurrently
- `(task1 | task2) >> task3` - Combined: parallel then sequential

**Decorators**:
- `@task` - Define a task
- `@task(inject_context=True)` - Task receives ExecutionContext

---

**Ready to build workflows? Start with `simple_pipeline.py`! 🚀**
