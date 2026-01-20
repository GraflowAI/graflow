# Graflow Workflow Patterns

## Table of Contents
- [Basic Task Definition](#basic-task-definition)
- [Workflow Context](#workflow-context)
- [Composition Operators](#composition-operators)
- [Channel Communication](#channel-communication)
- [Task Instance Binding](#task-instance-binding)
- [LLM Integration](#llm-integration)
- [Error Handling Policies](#error-handling-policies)

---

## Basic Task Definition

### Simple Task
```python
from graflow.core.decorators import task

@task
def hello():
    print("Hello, Graflow!")
    return "success"

# Execute directly
result = hello.run()
```

### Task with Parameters
```python
@task
def process_data(value: int, multiplier: int = 2) -> int:
    return value * multiplier

result = process_data.run(value=10, multiplier=3)  # Returns 30
```

### Task with Context Injection
```python
from graflow.core.context import TaskExecutionContext

@task(inject_context=True)
def context_task(ctx: TaskExecutionContext):
    print(f"Session: {ctx.session_id}")
    print(f"Task ID: {ctx.task_id}")
    channel = ctx.get_channel()
    channel.set("status", "running")
```

---

## Workflow Context

### Basic Workflow
```python
from graflow.core.workflow import workflow

with workflow("my_workflow") as wf:
    @task
    def step_a():
        return "a"

    @task
    def step_b():
        return "b"

    step_a >> step_b
    wf.execute("step_a")
```

### Workflow with Return Value
```python
with workflow("pipeline") as wf:
    # ... define tasks and graph ...
    result = wf.execute("entry_task")
    print(f"Final result: {result}")
```

### Workflow with Context Return
```python
with workflow("pipeline") as wf:
    # ... define tasks ...
    result, exec_ctx = wf.execute("entry_task", ret_context=True)
    final_channel = exec_ctx.get_channel()
    print(final_channel.get("accumulated_data"))
```

### Workflow with Initial Channel
```python
with workflow("pipeline") as wf:
    @task
    def process(config: dict):
        # config is auto-resolved from channel
        return config["value"] * 2

    wf.execute("process", initial_channel={"config": {"value": 10}})
```

---

## Composition Operators

### Sequential (`>>`)
```python
# Tasks execute one after another
fetch >> validate >> transform >> store
```

### Parallel (`|`)
```python
# Tasks execute concurrently
(task_a | task_b | task_c).set_group_name("parallel_tasks") >> aggregate
```

### Diamond Pattern (Fan-out + Fan-in)
```python
# source -> parallel transforms -> single sink
source >> (transform_a | transform_b) >> sink
```

### Multi-stage Pipeline
```python
# Multiple parallel stages
(load_a | load_b) >> validate >> (process_a | process_b) >> store
```

### Complex DAG
```python
a >> (b | c) >> d >> (e | f | g) >> h
```

---

## Channel Communication

### Basic Channel Operations
```python
@task(inject_context=True)
def producer(ctx: TaskExecutionContext):
    channel = ctx.get_channel()
    channel.set("data", [1, 2, 3])
    channel.set("config", {"batch_size": 100})

@task(inject_context=True)
def consumer(ctx: TaskExecutionContext):
    channel = ctx.get_channel()
    data = channel.get("data")        # [1, 2, 3]
    config = channel.get("config")    # {"batch_size": 100}
    keys = channel.keys()             # ["data", "config"]
```

### Channel Append
```python
@task(inject_context=True)
def accumulator(ctx: TaskExecutionContext):
    channel = ctx.get_channel()
    channel.append("results", "item1")
    channel.append("results", "item2")
    # channel.get("results") == ["item1", "item2"]
```

### Auto Keyword Resolution
```python
# Parameters matching channel keys are auto-resolved
@task(inject_context=True)
def setup(ctx: TaskExecutionContext):
    channel = ctx.get_channel()
    channel.set("user_name", "Alice")
    channel.set("count", 5)

@task
def greet(user_name: str, count: int = 1):
    # user_name="Alice", count=5 from channel
    for _ in range(count):
        print(f"Hello, {user_name}!")

# Disable auto-resolution if needed
@task(resolve_keyword_args=False)
def explicit_task(user_name: str = "default"):
    pass
```

### Parameter Priority
```
channel < bound < injection
```

1. **Channel** (lowest): Auto-resolved from `channel.set()`
2. **Bound**: Passed at task creation `task(param=value)`
3. **Injection** (highest): `inject_context`, `inject_llm_client`

---

## Task Instance Binding

### Multiple Instances from Single Task
```python
@task
def process(query: str) -> str:
    return f"Processing: {query}"

# Create separate instances with bound parameters
task1 = process(task_id="task1", query="Tokyo")
task2 = process(task_id="task2", query="Paris")

with workflow("multi_instance") as wf:
    task1 >> task2
    wf.execute("task1")
```

### Auto-generated Task ID
```python
# task_id is auto-generated if not provided
task1 = process(query="Tokyo")  # task_id: process_a3f2b9c1
task2 = process(query="Paris")  # task_id: process_b7e8f4d2
```

### Bound Parameters Override Channel
```python
with workflow("test") as wf:
    task = process(task_id="test", value=10)  # value=10 is bound
    wf.execute("test", initial_channel={"value": 100})
    # Uses value=10 (bound), not value=100 (channel)
```

---

## LLM Integration

### LLMClient Injection
```python
from graflow.llm.client import LLMClient

@task(inject_llm_client=True)
def summarize(llm_client: LLMClient, text: str):
    return llm_client.completion_text(
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
        max_tokens=100
    )
```

### PydanticAI Agent with Tools
```python
from pydantic_ai import RunContext
from graflow.llm.agents import PydanticLLMAgent, create_pydantic_ai_agent_with_litellm

# Create agent
pydantic_agent = create_pydantic_ai_agent_with_litellm(
    model="openai/gpt-4o-mini",
    system_prompt="You are a helpful assistant with tools."
)

# Register tools
@pydantic_agent.tool
def get_weather(ctx: RunContext[int], city: str) -> dict:
    return {"city": city, "temp": 20, "condition": "sunny"}

# Wrap for Graflow
agent = PydanticLLMAgent(pydantic_agent, name="weather_assistant")

@task(inject_context=True)
def ask(ctx: TaskExecutionContext, query: str):
    result = agent.run(query)
    return result["output"]
```

### Multi-turn Conversation
```python
@task(inject_context=True)
def chat(ctx: TaskExecutionContext, query: str, history: list | None = None):
    result = agent.run(query, message_history=history)

    # Store messages for next turn
    channel = ctx.get_channel()
    channel.set("messages", result["metadata"]["messages"])

    return result["output"]
```

---

## Error Handling Policies

### Strict Mode (Default)
```python
from graflow.core.task import ParallelGroup

# All tasks must succeed
group = ParallelGroup([task_a, task_b, task_c], name="strict_group")
# If any task fails, entire group fails
```

### Best Effort
```python
from graflow.coordination.executor import ExecutionPolicy

group = ParallelGroup(
    [task_a, task_b, task_c],
    name="best_effort",
    execution_policy=ExecutionPolicy.BEST_EFFORT
)
# Continues even if some tasks fail
```

### At Least N
```python
group = ParallelGroup(
    [task_a, task_b, task_c],
    name="at_least_2",
    execution_policy=ExecutionPolicy.AT_LEAST_N,
    min_successes=2
)
# Succeeds if at least 2 tasks complete
```

### Critical Tasks Only
```python
group = ParallelGroup(
    [task_a, task_b, task_c],
    name="critical",
    execution_policy=ExecutionPolicy.CRITICAL_ONLY,
    critical_tasks=["task_a"]  # Only task_a must succeed
)
```
