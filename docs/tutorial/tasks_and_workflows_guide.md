# Tasks and Workflows Guide

A practical guide to building workflows in Graflow — from your first task to advanced patterns.

This guide teaches you how to define tasks and build workflows through practical examples.

### Cheatsheet

| Concept | Syntax | Purpose |
|---------|--------|---------|
| Define task | `@task` | Convert function to task |
| Custom task ID | `@task(task_id="id")` | Set explicit task identifier |
| Create workflow | `with workflow("name") as wf:` | Define task graph |
| Sequential | `task_a >> task_b` | Run tasks in order |
| Parallel | `task_a \| task_b` | Run tasks concurrently |
| Chain tasks | `chain(task_a, task_b, task_c)` | Create sequential chain |
| Parallel tasks | `parallel(task_a, task_b, task_c)` | Create parallel group |
| Task instance | `task(task_id="id", param=value)` | Create a new task instance with parameters |
| Set group name | `task_group.set_group_name("name")` | Rename parallel group |
| Configure execution | `task_group.with_execution(policy="...")` | Set parallel group execution policy |
| Context injection | `@task(inject_context=True)` | Access channels/workflow control |
| LLM client injection | `@task(inject_llm_client=True)` | Direct LLM API calls |
| LLM agent injection | `@task(inject_llm_agent="name")` | Inject SuperAgent with tools |
| Get channel | `ctx.get_channel()` | Access key-value channel |
| Set with TTL | `channel.set(key, value, ttl=300)` | Store with expiration (seconds) |
| Append to list | `channel.append(key, value)` | Add to end of list |
| Prepend to list | `channel.prepend(key, value)` | Add to beginning of list |
| Get typed channel | `ctx.get_typed_channel(Schema)` | Type-safe channel access |
| Request feedback | `ctx.request_feedback(...)` | Human-in-the-loop approval/input |
| Initial params | `wf.execute(initial_channel={...})` | Set workflow parameters |
| Get all results | `wf.execute(ret_context=True)` | Access all task results |
| Get task result | `ctx.get_result(task_id)` | Retrieve specific task result |
| Enqueue task | `ctx.next_task(task)` | Add task to queue, continue normally |
| Jump to task | `ctx.next_task(task, goto=True)` | Jump to existing task, skip successors |
| Self-loop | `ctx.next_iteration()` | Retry/convergence pattern |
| Normal exit | `ctx.terminate_workflow()` | Exit successfully |
| Error exit | `ctx.cancel_workflow()` | Exit with error |

---

## Table of Contents

**Getting Started**
- [Level 1: Your First Task](#level-1-your-first-task) - @task decorator and task IDs
- [Level 2: Your First Workflow](#level-2-your-first-workflow) - Workflow context and execution
- [Level 3: Task Composition](#level-3-task-composition) - Sequential (>>) and parallel (|) operators
- [Level 4: Passing Parameters](#level-4-passing-parameters) - Channels and parameter binding

**Core Concepts**
- [Level 5: Task Instances](#level-5-task-instances) - Reusing tasks with different parameters
- [Level 6: Channels and Context](#level-6-channels-and-context) - Inter-task communication and injection
- [Level 7: Execution Patterns](#level-7-execution-patterns) - Getting results and controlling execution
- [Level 8: Complex Workflows](#level-8-complex-workflows) - Diamond patterns and multi-instance

**Advanced Topics**
- [Level 9: Dynamic Task Generation](#level-9-dynamic-task-generation) - Runtime task addition and control flow

**Reference**
- [Best Practices](#best-practices)
- [Summary](#summary)

---

## Core Concepts

Before we begin, here are the key concepts:

- **Task**: A unit of work (Python function with `@task` decorator)
- **Workflow**: A collection of tasks with dependencies
- **Task Graph**: The directed graph representing task execution order
- **Execution Context**: Runtime state (channels, results, metadata)

---

## Level 1: Your First Task

Let's start with the absolute basics — the `@task` decorator.

### The @task Decorator

Convert any Python function into a Graflow task:

```python
from graflow.core.decorators import task

@task
def hello():
    """A simple task."""
    print("Hello, Graflow!")
    return "success"
```

**What just happened?**
- `@task` converts a regular function into a Graflow task
- The task can be used in workflows or executed directly

### Custom Task IDs

By default, the function name becomes the task ID. You can specify a custom ID:

```python
# Default: task_id is "hello"
@task
def hello():
    print("Hello!")

# Custom: task_id is "greeting_task"
@task(task_id="greeting_task")
def hello():
    print("Hello!")
```

**💡 Key Takeaways:**
- Use `@task` to create tasks
- Default `task_id` is the function name
- Use `@task(task_id="custom_id")` for explicit naming

### Testing Tasks with .run()

Tasks can be executed directly using `.run()` for testing:

```python
@task
def calculate(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

# Test the task directly
result = calculate.run(x=5, y=3)
print(result)  # Output: 8
```

**When to use `.run()`:**
- ✅ Unit testing individual tasks
- ✅ Quick verification of task logic
- ✅ Debugging task behavior
- ❌ Not for production workflows (use `workflow.execute()`)

**Example: Testing with parameters**

```python
@task
def process_data(data: list[int], multiplier: int = 2) -> list[int]:
    """Process data with a multiplier."""
    return [x * multiplier for x in data]

# Test with different parameters
result1 = process_data.run(data=[1, 2, 3])
print(result1)  # Output: [2, 4, 6]

result2 = process_data.run(data=[1, 2, 3], multiplier=3)
print(result2)  # Output: [3, 6, 9]
```

**💡 Key Takeaway:** Use `.run()` to test tasks in isolation before using them in workflows.

---

## Level 2: Your First Workflow

Now let's connect multiple tasks together in a workflow.

### Complete Workflow Example

```python
from graflow.core.workflow import workflow
from graflow.core.decorators import task

with workflow("simple_pipeline") as wf:
    @task
    def start():
        print("Starting!")

    @task
    def middle():
        print("Middle!")

    @task
    def end():
        print("Ending!")

    # Connect tasks: start → middle → end
    start >> middle >> end

    # Execute the workflow
    wf.execute()
```

**Output:**
```
Starting!
Middle!
Ending!
```

**What's happening:**
- `with workflow("name")` creates a workflow context
- Tasks defined inside are automatically registered
- `>>` connects tasks sequentially (start → middle → end)
- `wf.execute()` runs the workflow

**💡 Key Takeaways:**
- Use `with workflow("name")` to create workflows
- Define tasks inside the workflow context
- Use `>>` to connect tasks sequentially
- Call `wf.execute()` to run the workflow

---

## Level 3: Task Composition

Learn how to combine tasks using `>>` (sequential) and `|` (parallel) operators.

### Combining Sequential and Parallel

```python
with workflow("composition") as wf:
    @task
    def start():
        print("Start")

    @task
    def parallel_a():
        print("Parallel A")

    @task
    def parallel_b():
        print("Parallel B")

    @task
    def end():
        print("End")

    # Pattern: start → (parallel_a | parallel_b) → end
    start >> (parallel_a | parallel_b) >> end

    wf.execute()
```

**Execution Flow:**
1. `start` runs first
2. `parallel_a` and `parallel_b` run concurrently
3. `end` runs after both parallel tasks finish

**Output:**
```
Start
Parallel A
Parallel B
End
```

**Operators:**
- `>>` creates sequential dependencies (run in order)
- `|` creates parallel execution (run concurrently)
- Use parentheses to group: `(task_a | task_b)`

**💡 Key Takeaways:**
- `task_a >> task_b` means "run a, then run b"
- `task_a | task_b` means "run a and b concurrently"
- Mix operators for complex patterns: `a >> (b | c) >> d`

### Helper Functions: chain() and parallel()

For creating sequences and groups with multiple tasks, use the helper functions:

```python
from graflow.core.task import chain, parallel

with workflow("helpers") as wf:
    @task
    def task_a():
        print("A")

    @task
    def task_b():
        print("B")

    @task
    def task_c():
        print("C")

    @task
    def task_d():
        print("D")

    # Using chain(*tasks) - equivalent to task_a >> task_b >> task_c
    seq = chain(task_a, task_b, task_c)

    # Using parallel(*tasks) - equivalent to task_a | task_b | task_c
    par = parallel(task_a, task_b, task_c)

    # Combine them
    _pipeline = seq >> par

    wf.execute()
```

**Function signatures:**
- `chain(*tasks)` - Takes 1 or more tasks as separate arguments
- `parallel(*tasks)` - Takes 2 or more tasks as separate arguments

**When to use:**
- `chain(*tasks)`: Cleaner when chaining 3+ tasks
- `parallel(*tasks)`: Cleaner when grouping 3+ tasks
- Operators (`>>`, `|`): More readable for 2 tasks or mixed patterns

**Example: Dynamic task lists**

```python
# If you have tasks in a list, unpack them with *
task_list = [task_a, task_b, task_c, task_d]

# Unpack the list into parallel()
parallel_group = parallel(*task_list)

# Or use operators in a loop
group = task_list[0]
for task in task_list[1:]:
    group = group | task
```

**Example: Using with bound arguments (task instances)**

```python
@task
def fetch_weather(city: str) -> dict:
    return {"city": city, "temp": 20}

# Create task instances with bound parameters
tokyo = fetch_weather(task_id="tokyo", city="Tokyo")
paris = fetch_weather(task_id="paris", city="Paris")
london = fetch_weather(task_id="london", city="London")

with workflow("weather") as wf:
    # Use parallel() with task instances
    all_cities = parallel(tokyo, paris, london)

    wf.execute()
```

**Example: Dynamic instances with chain() and parallel()**

```python
@task
def process_batch(batch_id: int, data: list) -> dict:
    return {"batch_id": batch_id, "count": len(data)}

# Generate task instances dynamically
cities = ["Tokyo", "Paris", "London", "NYC"]
fetch_tasks = [
    fetch_weather(task_id=f"fetch_{city.lower()}", city=city)
    for city in cities
]

batches = [1, 2, 3]
process_tasks = [
    process_batch(task_id=f"batch_{i}", batch_id=i, data=[])
    for i in batches
]

with workflow("dynamic") as wf:
    # Use parallel() with task instances
    all_fetches = parallel(*fetch_tasks)

    # Use chain() with task instances
    all_batches = chain(*process_tasks)

    # Combine
    all_fetches >> all_batches

    wf.execute()
```

### Configuring Parallel Groups

Parallel groups can be customized with names and execution policies:

```python
with workflow("configured") as wf:
    @task
    def task_a():
        print("A")

    @task
    def task_b():
        print("B")

    @task
    def task_c():
        print("C")

    # Create parallel group with custom name
    group = (task_a | task_b | task_c).set_group_name("my_parallel_tasks")

    # Configure execution policy
    group.with_execution(policy="best_effort")  # Continue even if some tasks fail

    wf.execute()
```

**Available execution policies:**

| Policy | Behavior |
|--------|----------|
| `"strict"` (default) | All tasks must succeed, fail if any fails |
| `"best_effort"` | Continue even if tasks fail, collect results |
| `AtLeastNGroupPolicy(min_success=N)` | At least N tasks must succeed |
| `CriticalGroupPolicy(critical_task_ids=[...])` | Specific tasks must succeed |

**Example: Best-effort parallel execution**

```python
# Continue workflow even if some parallel tasks fail
(fetch_api | fetch_db | fetch_cache).with_execution(policy="best_effort")
```

**Example: Custom group name**

```python
# Rename group for clarity in logs and visualization
parallel_fetches = (fetch_a | fetch_b | fetch_c).set_group_name("data_fetches")
```

**Example: Advanced execution configuration**

```python
from graflow.coordination.coordinator import CoordinationBackend

# Use threading backend with custom thread count
(task_a | task_b | task_c | task_d).with_execution(
    backend=CoordinationBackend.THREADING,
    backend_config={"thread_count": 2},
    policy="best_effort"
)

# AtLeastN policy: Require at least 3 out of 4 tasks to succeed
from graflow.core.handlers.group_policy import AtLeastNGroupPolicy

(task_a | task_b | task_c | task_d).with_execution(
    policy=AtLeastNGroupPolicy(min_success=3)
)

# Critical policy: Specific tasks must succeed
from graflow.core.handlers.group_policy import CriticalGroupPolicy

(task_a | task_b | task_c).with_execution(
    policy=CriticalGroupPolicy(critical_task_ids=["task_a", "task_b"])
)
```

**💡 Key Takeaways:**
- Use `chain()` and `parallel()` for cleaner multi-task creation
- Use `.set_group_name()` to give parallel groups meaningful names
- Use `.with_execution(policy=...)` to control failure handling
- Configure `backend` and `backend_config` for execution control

---

## Level 4: Passing Parameters

Learn how to pass data between tasks using channels and parameter binding.

### Using Channels for Inter-Task Communication

Tasks communicate by reading and writing to a shared channel (detailed in [Level 6](#level-6-channels-and-context)):

```python
from graflow.core.context import TaskExecutionContext

with workflow("channel_communication") as wf:
    @task(inject_context=True)
    def producer(ctx: TaskExecutionContext):
        channel = ctx.get_channel()
        channel.set("user_id", "user_123")

    @task(inject_context=True)
    def consumer(ctx: TaskExecutionContext):
        channel = ctx.get_channel()
        user_id = channel.get("user_id")
        print(f"User: {user_id}")

    producer >> consumer
    wf.execute()
```

### Partial Parameter Binding

You can bind some parameters at task creation, while others come from the channel:

```python
with workflow("partial_binding") as wf:
    @task
    def calculate(base: int, multiplier: int, offset: int) -> int:
        result = base * multiplier + offset
        print(f"calculate: {base} * {multiplier} + {offset} = {result}")
        return result

    # Bind only 'base', others come from channel
    task_instance = calculate(task_id="calc", base=10)

    # Execute with channel values for multiplier and offset
    _, ctx = wf.execute(
        ret_context=True,
        initial_channel={"multiplier": 3, "offset": 5}
    )

    result = ctx.get_result("calc")
    print(f"Result: {result}")
```

**Output:**
```
calculate: 10 * 3 + 5 = 35
Result: 35
```

**What happened:**
- `base=10` is bound at task creation (takes priority)
- `multiplier=3` and `offset=5` come from channel
- Bound parameters always override channel values

**💡 Key Takeaways:**
- Tasks can communicate via channels (see [Level 6](#level-6-channels-and-context) for details)
- Bind some parameters at task creation, let others come from channel
- Bound parameters have higher priority than channel values

---

## Level 5: Task Instances

**New in Graflow**: Create multiple instances from one task definition.

### The Problem

You want to reuse the same task logic with different parameters:

```python
# ❌ Without task instances (repetitive)
@task
def fetch_tokyo():
    return fetch("Tokyo")

@task
def fetch_paris():
    return fetch("Paris")
```

### The Solution

Create task instances with bound parameters:

```python
# ✅ With task instances (reusable)
@task
def fetch_weather(city: str) -> str:
    return f"Weather for {city}"

# Create instances with different parameters
tokyo = fetch_weather(task_id="tokyo", city="Tokyo")
paris = fetch_weather(task_id="paris", city="Paris")
london = fetch_weather(task_id="london", city="London")

with workflow("weather") as wf:
    # Use instances in workflow
    tokyo >> paris >> london
    wf.execute()
```

**Output:**
```
Weather for Tokyo
Weather for Paris
Weather for London
```

### Auto-Generated Task IDs

Don't want to name every task? Omit `task_id`:

```python
@task
def process(value: int) -> int:
    return value * 2

# Auto-generated IDs: process_{random_uuid}
task1 = process(value=10)  # task_id: process_a3f2b9c1
task2 = process(value=20)  # task_id: process_b7e8f4d2
task3 = process(value=30)  # task_id: process_c5d9e6f7

with workflow("auto_ids") as wf:
    task1 >> task2 >> task3
    wf.execute()
```

**⚠️ Caution: Ensure Unique Task IDs**

When creating multiple task instances, make sure each has a unique `task_id`:

```python
# ✅ Good: Unique task_ids
tokyo = fetch_weather(task_id="tokyo", city="Tokyo")
paris = fetch_weather(task_id="paris", city="Paris")
london = fetch_weather(task_id="london", city="London")

# ❌ Bad: Duplicate task_ids cause conflicts
task1 = fetch_weather(task_id="fetch", city="Tokyo")
task2 = fetch_weather(task_id="fetch", city="Paris")  # ERROR: "fetch" already exists!

# ✅ Good: Auto-generated IDs are always unique
task1 = fetch_weather(city="Tokyo")   # Auto: fetch_weather_a3f2b9c1
task2 = fetch_weather(city="Paris")   # Auto: fetch_weather_b7e8f4d2
```

**💡 Key Takeaways:**
- Task instances reuse task logic with different parameters
- Specify `task_id` for named instances (must be unique!)
- Omit `task_id` for auto-generated IDs (guaranteed unique)
- Each instance is independent

---

## Level 6: Channels and Context

### Channel Backends

Graflow supports two backends for seamless local-to-distributed transition:

**1. MemoryChannel (Default)** - For local execution:
- ✅ Fast: In-memory, minimal latency
- ✅ Simple: No infrastructure required
- ✅ Checkpoint-compatible: Auto-saved with checkpoints
- ⚠️ Limitation: Single process only

**2. RedisChannel** - For distributed execution:
- ✅ Distributed: Share state across workers/machines
- ✅ Persistent: Redis persistence for fault tolerance
- ✅ Scalable: Consistent data across many workers
- ⚠️ Requires: Redis server

**Switching backends:**

```python
# Local execution (default) - uses MemoryChannel
with workflow("local") as wf:
    task_a >> task_b
    wf.execute()

# Distributed execution - uses RedisChannel
from graflow.channels.factory import ChannelFactory, ChannelBackend

channel = ChannelFactory.create_channel(
    backend=ChannelBackend.REDIS,
    redis_client=redis_client
)

with workflow("distributed") as wf:
    task_a >> task_b
    wf.execute()
```

### Working with Channels

#### Basic Channel: `ctx.get_channel()`

Access the basic channel for simple key-value storage:

```python
@task(inject_context=True)
def producer(ctx: TaskExecutionContext):
    """Write data to channel."""
    channel = ctx.get_channel()

    # Store simple values
    channel.set("user_id", "user_123")
    channel.set("score", 95.5)
    channel.set("active", True)

    # Store complex objects
    channel.set("user_profile", {
        "name": "Alice",
        "email": "alice@example.com",
        "age": 30
    })

@task(inject_context=True)
def consumer(ctx: TaskExecutionContext):
    """Read data from channel."""
    channel = ctx.get_channel()

    # Retrieve values
    user_id = channel.get("user_id")        # "user_123"
    score = channel.get("score")            # 95.5
    active = channel.get("active")          # True
    profile = channel.get("user_profile")   # dict

    # With default value
    setting = channel.get("setting", default="default_value")
```

**Channel Methods:**

| Method | Description | Example |
|--------|-------------|---------|
| `set(key, value)` | Store a value | `channel.set("count", 42)` |
| `set(key, value, ttl)` | Store with expiration (seconds) | `channel.set("temp", 100, ttl=300)` |
| `get(key)` | Retrieve a value | `value = channel.get("count")` |
| `get(key, default)` | Retrieve with fallback | `value = channel.get("count", default=0)` |
| `append(key, value)` | Append to list | `channel.append("logs", "entry")` |
| `append(key, value, ttl)` | Append with expiration | `channel.append("logs", "entry", ttl=60)` |
| `prepend(key, value)` | Prepend to list | `channel.prepend("queue", "item")` |
| `delete(key)` | Remove a key | `channel.delete("count")` |
| `exists(key)` | Check if key exists | `if channel.exists("count"):` |

**List Operations: append() and prepend()**

Channels support list operations for collecting multiple values:

```python
@task(inject_context=True)
def collect_logs(ctx: TaskExecutionContext):
    channel = ctx.get_channel()

    # Append to end of list (FIFO queue)
    channel.append("logs", "Log entry 1")
    channel.append("logs", "Log entry 2")
    channel.append("logs", "Log entry 3")

    logs = channel.get("logs")
    print(logs)  # ["Log entry 1", "Log entry 2", "Log entry 3"]

@task(inject_context=True)
def use_stack(ctx: TaskExecutionContext):
    channel = ctx.get_channel()

    # Prepend to beginning of list (LIFO stack)
    channel.prepend("stack", "First")
    channel.prepend("stack", "Second")
    channel.prepend("stack", "Third")

    stack = channel.get("stack")
    print(stack)  # ["Third", "Second", "First"]
```

**Use cases:**
- `append()`: Build logs, collect results from parallel tasks, FIFO queues
- `prepend()`: LIFO stacks, priority items, reverse-order collection

**Time-to-Live (TTL): Automatic Expiration**

Use TTL to automatically expire temporary data:

```python
@task(inject_context=True)
def cache_data(ctx: TaskExecutionContext):
    channel = ctx.get_channel()

    # Cache for 5 minutes (300 seconds)
    channel.set("api_response", {"data": "..."}, ttl=300)

    # Temporary flag expires in 60 seconds
    channel.set("processing", True, ttl=60)

    # Collect logs that expire after 10 minutes
    channel.append("recent_logs", "Error occurred", ttl=600)

@task(inject_context=True)
def check_cache(ctx: TaskExecutionContext):
    channel = ctx.get_channel()

    # After TTL expires, key is automatically removed
    data = channel.get("api_response", default="expired")
    if data == "expired":
        print("Cache expired, refetching...")
```

**TTL Behavior:**
- TTL is in **seconds**
- Key expires and is automatically deleted after TTL
- Calling `get()` on expired key returns `None` (or default value)
- Both `set()` and `append()`/`prepend()` support TTL
- Useful for temporary caches, rate limiting, session data

**Example: Collecting parallel task results with TTL**

```python
@task(inject_context=True)
def fetch_data(ctx: TaskExecutionContext, source: str):
    channel = ctx.get_channel()
    data = f"Data from {source}"

    # Collect results with 1-hour expiration
    channel.append("fetch_results", data, ttl=3600)

    return data

with workflow("collect_results") as wf:
    fetch_a = fetch_data(task_id="fetch_a", source="api")
    fetch_b = fetch_data(task_id="fetch_b", source="db")
    fetch_c = fetch_data(task_id="fetch_c", source="cache")

    parallel(fetch_a, fetch_b, fetch_c)

    wf.execute()
```

#### Type-Safe Channel: `ctx.get_typed_channel()`

Use typed channels for compile-time type checking and IDE autocomplete:

```python
from typing import TypedDict

# Define schema
class UserProfile(TypedDict):
    user_id: str
    name: str
    email: str
    age: int
    premium: bool

@task(inject_context=True)
def collect_user_data(ctx: TaskExecutionContext):
    """Store user data with type safety."""

    # Get typed channel
    typed_channel = ctx.get_typed_channel(UserProfile)

    # IDE autocompletes fields!
    user_profile: UserProfile = {
        "user_id": "user_123",
        "name": "Alice",
        "email": "alice@example.com",
        "age": 30,
        "premium": True
    }

    # Type-checked storage
    typed_channel.set("current_user", user_profile)

@task(inject_context=True)
def process_user_data(ctx: TaskExecutionContext):
    """Retrieve user data with type safety."""

    # Get typed channel with same schema
    typed_channel = ctx.get_typed_channel(UserProfile)

    # Retrieve with type hints
    user: UserProfile = typed_channel.get("current_user")

    # IDE knows the structure!
    print(user["name"])    # IDE autocompletes "name"
    print(user["email"])   # IDE autocompletes "email"
```

**Benefits of Typed Channels:**

- ✅ **IDE Autocomplete**: Field names and types suggested
- ✅ **Type Checking**: mypy/pyright catches type errors
- ✅ **Self-Documenting**: TypedDict serves as API contract
- ✅ **Refactoring Safety**: Rename fields with IDE support
- ✅ **Team Collaboration**: Shared schemas prevent mistakes

**When to Use Each:**

| Use Case | Method | Why |
|----------|--------|-----|
| Simple values (strings, numbers) | `get_channel()` | Less overhead |
| Ad-hoc data exchange | `get_channel()` | No schema needed |
| Structured data | `get_typed_channel()` | Type safety |
| Team collaboration | `get_typed_channel()` | Shared schema |
| Large projects | `get_typed_channel()` | Maintainability |

**Example: Mixed Usage**

```python
@task(inject_context=True)
def process_order(ctx: TaskExecutionContext):
    # Use typed channel for structured data
    order_channel = ctx.get_typed_channel(OrderData)
    order = order_channel.get("current_order")

    # Use basic channel for simple flags
    basic_channel = ctx.get_channel()
    basic_channel.set("processing_started", True)
    basic_channel.set("timestamp", "2024-01-01T12:00:00")
```

### Dependency Injection

Graflow provides three types of dependency injection to automatically provide resources to tasks.

#### 1. Context Injection: `inject_context=True`

Injects the execution context to access channels, results, and workflow metadata:

```python
@task(inject_context=True)
def my_task(ctx: TaskExecutionContext, value: int):
    # Access channel
    channel = ctx.get_channel()
    channel.set("result", value * 2)

    # Access session info
    print(f"Session: {ctx.session_id}")

    # Access other task results
    previous = ctx.get_result("previous_task")

    return value * 2
```

**When to use:**
- Inter-task communication via channels
- Access results from other tasks
- Control workflow (next_task, next_iteration, terminate_workflow)

#### 2. LLM Client Injection: `inject_llm_client=True`

Injects a lightweight LLM client for direct API calls:

```python
from graflow.llm.client import LLMClient

@task(inject_llm_client=True)
def analyze_text(llm: LLMClient, text: str) -> str:
    # Direct LLM API call
    response = llm.completion_text(
        messages=[{"role": "user", "content": f"Analyze: {text}"}],
        model="gpt-4o-mini"
    )
    return response
```

**When to use:**
- Simple LLM API calls without agent loops
- Multi-model scenarios (use different models per task)
- Cost optimization (cheap model for simple tasks)

**Supports:** OpenAI ChatGPT, Anthropic Claude, Google Gemini, AWS Bedrock, and more via LiteLLM.

#### 3. LLM Agent Injection: `inject_llm_agent="agent_name"`

Injects a full-featured LLM agent (SuperAgent) with ReAct loops and tools:

```python
from graflow.llm.agents.base import LLMAgent

# First, register the agent in workflow
context.register_llm_agent("supervisor", my_agent)

# Then inject into task
@task(inject_llm_agent="supervisor")
def supervise_task(agent: LLMAgent, query: str) -> str:
    # Agent handles ReAct loop, tool calls internally
    result = agent.run(query)
    return result["output"]
```

**When to use:**
- Complex reasoning tasks requiring tool calls
- Multi-turn conversations
- Tasks needing autonomous agent behavior

**Compatible with:** Google ADK, PydanticAI, custom agents

#### Injection Summary

| Injection Type | Parameter | Use Case |
|----------------|-----------|----------|
| `inject_context=True` | `ctx: TaskExecutionContext` | Channels, workflow control, results |
| `inject_llm_client=True` | `llm: LLMClient` | Simple LLM API calls |
| `inject_llm_agent="name"` | `agent: LLMAgent` | Complex agent tasks with tools |

**💡 Key Points:**
- Injections happen automatically at task execution
- First parameter receives the injected dependency
- Can combine: `inject_context=True, inject_llm_client=True`
- Agent must be registered before use: `context.register_llm_agent(name, agent)`

#### Alternative: Accessing LLM Client/Agent via Context

When using `inject_context=True`, you can also access LLM client and agents through the context instead of direct injection:

```python
@task(inject_context=True)
def task_with_llm(ctx: TaskExecutionContext, query: str):
    # Access LLM client via context
    response = ctx.llm_client.completion_text(
        messages=[{"role": "user", "content": query}],
        model="gpt-4o-mini"
    )

    # Access LLM agent via context
    agent = ctx.get_llm_agent("supervisor")
    result = agent.run(query)

    return {"llm": response, "agent": result}
```

**When to use:**
- Direct injection (`inject_llm_client=True`): Cleaner when only using LLM
- Via context (`ctx.llm_client`): When you also need channels or workflow control

### Human-in-the-Loop: `ctx.request_feedback()`

Request human feedback during workflow execution using `ctx.request_feedback()`:

```python
@task(inject_context=True)
def request_approval(ctx: TaskExecutionContext, deployment_plan: dict) -> bool:
    """Request human approval before deployment."""

    response = ctx.request_feedback(
        feedback_type="approval",
        prompt="Approve deployment to production?",
        timeout=300,  # Wait 5 minutes
        notification_config={
            "type": "slack",
            "webhook_url": "https://hooks.slack.com/services/XXX",
            "message": "Deployment approval needed!"
        }
    )

    if not response.approved:
        ctx.cancel_workflow("Deployment rejected by user")

    return response.approved
```

**Feedback Types:**

1. **Approval** - Yes/No decision
   ```python
   response = ctx.request_feedback(
       feedback_type="approval",
       prompt="Approve this action?"
   )
   # response.approved: bool
   ```

2. **Text Input** - Free-form text
   ```python
   response = ctx.request_feedback(
       feedback_type="text",
       prompt="Enter configuration value:"
   )
   # response.text: str
   ```

3. **Selection** - Choose one option
   ```python
   response = ctx.request_feedback(
       feedback_type="selection",
       prompt="Choose deployment environment:",
       options=["staging", "production"]
   )
   # response.selected: str
   ```

4. **Multi-Selection** - Choose multiple options
   ```python
   response = ctx.request_feedback(
       feedback_type="multi_selection",
       prompt="Select features to enable:",
       options=["feature_a", "feature_b", "feature_c"]
   )
   # response.selected: list[str]
   ```

**Timeout and Checkpoint Behavior:**

When timeout occurs, Graflow automatically creates a checkpoint and pauses the workflow:

```python
response = ctx.request_feedback(
    feedback_type="approval",
    prompt="Approve deployment?",
    timeout=300  # 5 minutes
)

# If no response within 5 minutes:
# 1. Checkpoint is automatically created
# 2. Workflow pauses
# 3. User can provide feedback later via API
# 4. Workflow resumes from checkpoint when feedback is received
```

**Use Cases:**
- Deployment approvals
- Data validation reviews
- Parameter tuning by domain experts
- Error recovery decisions

### Idempotence with Request Feedback

**Important for HITL workflows**: When using `ctx.request_feedback()`, tasks must be idempotent because they may resume from a checkpoint after receiving feedback.

**Why Idempotence Matters for Request Feedback:**

When a task requests feedback and times out:
1. Checkpoint is automatically created
2. Workflow pauses
3. User provides feedback later
4. **Workflow resumes from checkpoint and re-executes the task**

This means the task may run multiple times, so it must be safe to re-execute:

```python
# ⚠️ NOT Idempotent - Dangerous with request_feedback
@task(inject_context=True)
def deploy_with_approval(ctx: TaskExecutionContext):
    # Deploy FIRST (wrong order!)
    deployment_id = api.deploy_to_production()

    # Then ask for approval
    response = ctx.request_feedback(
        feedback_type="approval",
        prompt="Approve deployment?"
    )

    # If timeout occurs and task resumes, deploy happens AGAIN!
    # This creates duplicate deployments!
```

```python
# ✅ Idempotent - Safe with request_feedback
@task(inject_context=True)
def deploy_with_approval(ctx: TaskExecutionContext, deployment_plan: dict):
    channel = ctx.get_channel()

    # Check if already deployed
    if not channel.get("deployment_approved"):
        # Ask for approval FIRST
        response = ctx.request_feedback(
            feedback_type="approval",
            prompt="Approve deployment?",
            timeout=300
        )

        if not response.approved:
            ctx.cancel_workflow("Deployment rejected")

        # Mark as approved
        channel.set("deployment_approved", True)

    # Check if already deployed
    if not channel.get("deployment_completed"):
        # Deploy only once
        deployment_id = api.deploy_to_production(deployment_plan)
        channel.set("deployment_completed", True)
        channel.set("deployment_id", deployment_id)

    return channel.get("deployment_id")
```

**Best Practices:**

1. **Request feedback BEFORE side effects**
2. **Use channel flags** to track completion state
3. **Check flags before performing actions** to prevent duplicates
4. **Use idempotency keys** for external API calls

**💡 Key Takeaway:** Always make tasks idempotent when using `ctx.request_feedback()` to safely handle checkpoint resume.

### Parameter Priority

When resolving parameters: **Injection > Bound > Channel**

```python
@task
def calculate(value: int, multiplier: int) -> int:
    return value * multiplier

# Bind value=10, multiplier from channel
task = calculate(task_id="calc", value=10)

wf.execute(initial_channel={"value": 100, "multiplier": 5})
# Result: 10 × 5 = 50 (bound value beats channel value)
```

---

## Level 7: Execution Patterns

### Understanding Task Results

When tasks return values, Graflow stores them in the channel using the task's `task_id`:

```python
# Auto-generated task_id (function name)
@task
def calculate():
    return 42

# Stored as: channel.set("calculate.__result__", 42)
# Access: ctx.get_result("calculate") → 42

# Custom task_id
task1 = calculate(task_id="calc1")
task2 = calculate(task_id="calc2")

# Stored as: channel.set("calc1.__result__", 42)
#            channel.set("calc2.__result__", 42)
# Access: ctx.get_result("calc1"), ctx.get_result("calc2")
```

**Result storage format:** `{task_id}.__result__`

```python
# When a task completes:
channel.set(f"{task_id}.__result__", return_value)

# When you call get_result():
def get_result(task_id: str, default=None):
    return channel.get(f"{task_id}.__result__", default)
```

### Pattern 1: Get Final Result

```python
with workflow("simple") as wf:
    @task
    def compute():
        return 42

    result = wf.execute()
    print(result)  # 42 (last task's return value)
```

### Pattern 2: Get All Results

Get results from all tasks using execution context:

```python
with workflow("all_results") as wf:
    @task
    def task_a():
        return "A"

    @task
    def task_b():
        return "B"

    task_a >> task_b

    # Get execution context to access all results
    _, ctx = wf.execute(ret_context=True)

    # Access individual task results
    print(ctx.get_result("task_a"))  # Output: A
    print(ctx.get_result("task_b"))  # Output: B
```

**Key Points:**
- `ret_context=True` returns tuple: `(final_result, execution_context)`
- Use `ctx.get_result(task_id)` to get any task's result
- Results are automatically stored when tasks return values

### Pattern 3: Start from Specific Task

**Auto-Detection (No argument):**

When you call `wf.execute()` without arguments, Graflow automatically finds the start node:

```python
with workflow("auto_start") as wf:
    @task
    def step1():
        print("Step 1")

    @task
    def step2():
        print("Step 2")

    step1 >> step2

    # Auto-detects step1 (node with no predecessors)
    wf.execute()
```

**How auto-detection works:**
1. Finds all nodes with **no incoming edges** (no predecessors)
2. If **exactly one** node found → use it as start node
3. If **none** found → raises `GraphCompilationError` (cyclic graph or empty)
4. If **multiple** found → raises `GraphCompilationError` (ambiguous start point)

**Example: Multiple entry points (error)**

```python
with workflow("ambiguous") as wf:
    @task
    def task_a():
        print("A")

    @task
    def task_b():
        print("B")

    @task
    def task_c():
        print("C")

    # Two separate chains - two entry points!
    task_a >> task_c
    task_b >> task_c

    # ERROR: Multiple start nodes found (task_a and task_b)
    # wf.execute()  # Raises GraphCompilationError

    # Solution: Specify start node explicitly
    wf.execute(start_node="task_a")
```

**Manual start node:**

Skip earlier tasks by specifying a start node:

```python
with workflow("skip") as wf:
    @task
    def step1():
        print("Step 1")

    @task
    def step2():
        print("Step 2")

    @task
    def step3():
        print("Step 3")

    step1 >> step2 >> step3

    # Start from step2 (skip step1)
    wf.execute(start_node="step2")
```

**Output:**
```
Step 2
Step 3
```

**💡 Key Takeaways:**
- `wf.execute()` auto-detects start node (node with no predecessors)
- Raises error if zero or multiple start nodes found
- `wf.execute(start_node="task_id")` explicitly sets start point
- `wf.execute(ret_context=True)` returns `(result, context)`
- Use `ctx.get_result(task_id)` to get specific task results

---

## Level 8: Complex Workflows

### Diamond Pattern

One task splits into parallel branches, then merges:

```python
@task(inject_context=True)
def source(ctx: TaskExecutionContext, value: int) -> int:
    ctx.get_channel().set("value", value)
    return value

@task(inject_context=True)
def double(ctx: TaskExecutionContext) -> int:
    value = ctx.get_channel().get("value")
    result = value * 2
    ctx.get_channel().set("doubled", result)
    return result

@task(inject_context=True)
def triple(ctx: TaskExecutionContext) -> int:
    value = ctx.get_channel().get("value")
    result = value * 3
    ctx.get_channel().set("tripled", result)
    return result

@task(inject_context=True)
def combine(ctx: TaskExecutionContext) -> int:
    doubled = ctx.get_channel().get("doubled")
    tripled = ctx.get_channel().get("tripled")
    return doubled + tripled

with workflow("diamond") as wf:
    src = source(task_id="src", value=5)

    # Diamond: src → (double | triple) → combine
    src >> (double | triple) >> combine

    result = wf.execute(start_node="src")
    print(result)  # Output: 25 (5*2 + 5*3)
```

### Multi-Instance Pipeline

Process multiple items in parallel:

```python
@task
def fetch(source: str) -> dict:
    return {"source": source, "data": f"data_{source}"}

@task
def process(data: dict) -> str:
    return f"Processed {data['source']}"

with workflow("multi_pipeline") as wf:
    # Create instances
    fetch_a = fetch(task_id="fetch_a", source="api")
    fetch_b = fetch(task_id="fetch_b", source="db")
    fetch_c = fetch(task_id="fetch_c", source="file")

    # Run in parallel
    all_fetches = fetch_a | fetch_b | fetch_c

    _, ctx = wf.execute(
        start_node=all_fetches.task_id,
        ret_context=True
    )

    # Get results
    for task_id in ["fetch_a", "fetch_b", "fetch_c"]:
        print(ctx.get_result(task_id))
```

**💡 Key Pattern:** Create task instances → Combine with `|` → Execute in parallel

---

## Level 9: Dynamic Task Generation

**Advanced Feature**: Modify the workflow graph during execution.

### Why Runtime Dynamics?

**The Problem with Compile-Time Graphs:**

Many workflow systems require defining all branches and loops at compile time:

```python
# ❌ Compile-time approach (LangGraph style)
def should_retry(state):
    return "retry" if state["score"] < 0.8 else "continue"

graph.add_conditional_edges(
    "process",
    should_retry,  # All paths predefined
    {
        "retry": "retry_node",
        "continue": "finalize_node"
    }
)
app = graph.compile()  # Graph is now fixed
```

**Limitations:**
- All possible paths must be defined upfront
- Hard to handle dynamic conditions (file count, data size)
- Loop counts fixed at definition time
- Complex to express adaptive logic

**Graflow's Solution: Runtime Flexibility**

With Graflow, you write normal Python conditionals and generate tasks on-the-fly:

### Adding Tasks at Runtime

Use `context.next_task()` to add tasks dynamically or jump to existing tasks:

**The `goto` parameter:**

- **`ctx.next_task(task, goto=False)`** (default):
  - Enqueue the task to the execution queue
  - Continue to normal successors after current task completes
  - Use for adding additional work without changing control flow

- **`ctx.next_task(task, goto=True)`**:
  - Jump to the specified task immediately
  - Skip normal successors of current task
  - **Designed for jumping to existing tasks already in the graph**

**Example 1: Jumping to existing tasks in the graph**

Use `goto=True` to jump to tasks that were already defined in the workflow:

```python
with workflow("error_handling") as wf:
    @task(inject_context=True)
    def risky_operation(ctx: TaskExecutionContext):
        """Process data with potential errors."""
        try:
            # Risky operation
            if random.random() < 0.3:  # 30% chance of critical error
                raise CriticalError("Critical failure!")
            print("Operation succeeded")
        except CriticalError:
            # Jump to existing emergency handler task
            emergency_task = ctx.graph.get_node("emergency_handler")
            ctx.next_task(emergency_task, goto=True)  # Skip normal successors

    @task
    def emergency_handler():
        """Handle emergency situations."""
        print("Emergency handler activated!")
        # Send alerts, rollback, etc.

    @task
    def normal_continuation():
        """This runs only if risky_operation succeeds."""
        print("Continuing normal flow")

    # Define workflow
    risky_operation >> normal_continuation

    wf.execute()
```

**Output (on error):**
```
Emergency handler activated!
```

**Output (on success):**
```
Operation succeeded
Continuing normal flow
```

**Example 2: Conditional branching to existing tasks**

```python
with workflow("conditional") as wf:
    @task(inject_context=True)
    def router(ctx: TaskExecutionContext, user_type: str):
        """Route to different paths based on user type."""
        if user_type == "premium":
            premium_task = ctx.graph.get_node("premium_flow")
            ctx.next_task(premium_task, goto=True)
        elif user_type == "basic":
            basic_task = ctx.graph.get_node("basic_flow")
            ctx.next_task(basic_task, goto=True)

    @task
    def premium_flow():
        print("Premium user processing")

    @task
    def basic_flow():
        print("Basic user processing")

    @task
    def default_continuation():
        print("This is skipped when goto=True")

    router >> default_continuation

    wf.execute(initial_channel={"user_type": "premium"})
```

**Example 3: Enqueue additional work (goto=False)**

Use `goto=False` (default) to add tasks without changing control flow:

```python
@task(inject_context=True)
def process(ctx: TaskExecutionContext):
    @task
    def extra_logging():
        print("Extra logging task")

    # Enqueue: Add extra_logging, then continue to normal successors
    ctx.next_task(extra_logging)  # goto=False is default

    print("Main processing")

@task
def continuation():
    print("Normal continuation")

with workflow("enqueue_demo") as wf:
    process >> continuation
    wf.execute()
```

**Output:**
```
Main processing
Extra logging task
Normal continuation
```

**💡 Key Differences:**
- **`goto=False`** (default): "Do this task AND continue normally"
- **`goto=True`**: "Jump to this existing task INSTEAD of continuing normally"
- Use `ctx.graph.get_node(task_id)` to get existing tasks from the graph

### Self-Looping with next_iteration

Use `context.next_iteration()` for retry/convergence patterns:

```python
@task(inject_context=True)
def optimize(ctx: TaskExecutionContext):
    """Optimize until convergence."""
    channel = ctx.get_channel()
    iteration = channel.get("iteration", default=0)
    accuracy = channel.get("accuracy", default=0.5)

    # Training step
    new_accuracy = train_step(accuracy)
    print(f"Iteration {iteration}: accuracy={new_accuracy:.2f}")

    if new_accuracy >= 0.95:
        # Converged!
        print("Converged!")
        channel.set("final_accuracy", new_accuracy)
    else:
        # Continue iterating
        channel.set("iteration", iteration + 1)
        channel.set("accuracy", new_accuracy)
        ctx.next_iteration()

with workflow("optimization") as wf:
    wf.execute()
```

**Example Output:**
```
Iteration 0: accuracy=0.65
Iteration 1: accuracy=0.78
Iteration 2: accuracy=0.88
Iteration 3: accuracy=0.96
Converged!
```

**💡 Key Use Cases:**
- Retry logic with max attempts
- ML hyperparameter tuning
- Convergence-based algorithms
- Progressive enhancement

### Early Termination

#### Normal Termination: terminate_workflow

Use when you want to exit successfully:

```python
@task(inject_context=True)
def check_cache(ctx: TaskExecutionContext, key: str):
    """Check cache before processing."""
    cached = get_from_cache(key)

    if cached is not None:
        # Cache hit - no need to continue
        print(f"Cache hit: {cached}")
        ctx.terminate_workflow("Data found in cache")
        return cached

    # Cache miss - continue to next tasks
    print("Cache miss, proceeding...")
    return None

@task
def expensive_processing():
    """This won't run if cache hits."""
    print("Expensive processing...")
    return "processed"

with workflow("caching") as wf:
    check_cache(task_id="cache", key="my_key") >> expensive_processing
    wf.execute()
```

**With cache hit:**
```
Cache hit: cached_value
```

**With cache miss:**
```
Cache miss, proceeding...
Expensive processing...
```

#### Abnormal Termination: cancel_workflow

Use when an error occurs:

```python
@task(inject_context=True)
def validate_data(ctx: TaskExecutionContext, data: dict):
    """Validate data before processing."""
    if not data.get("valid"):
        # Invalid data - cancel entire workflow
        ctx.cancel_workflow("Data validation failed")

    return data

@task
def process_data(data: dict):
    print("Processing data...")
    return data

with workflow("validation") as wf:
    validate = validate_data(task_id="validate", data={"valid": False})
    validate >> process_data

    try:
        wf.execute()
    except Exception as e:
        print(f"Workflow canceled: {e}")
```

**Output:**
```
Workflow canceled: Data validation failed
```

**Differences:**

| Method | Task Completes? | Successors Run? | Error Raised? |
|--------|----------------|----------------|---------------|
| `terminate_workflow` | ✅ Yes | ❌ No | ❌ No |
| `cancel_workflow` | ❌ No | ❌ No | ✅ Yes (GraflowWorkflowCanceledError) |

**💡 Key Takeaways:**
- `next_task(task)` enqueues task and continues to normal successors
- `next_task(task, goto=True)` jumps to task, skipping normal successors
- `next_iteration()` creates self-loops for retry/convergence
- `terminate_workflow()` exits successfully
- `cancel_workflow()` exits with error

---

## Best Practices

### 1. Use Task Instances for Reusability

```python
# ✅ Good - Reusable task definition
@task
def fetch_data(source: str):
    return fetch(source)

api = fetch_data(task_id="api", source="api")
db = fetch_data(task_id="db", source="database")

# ❌ Avoid - Duplicated definitions
@task
def fetch_api():
    return fetch("api")

@task
def fetch_db():
    return fetch("database")
```

### 2. Always Use Type Hints

```python
# ✅ Good
@task
def process(value: int, multiplier: int = 2) -> int:
    return value * multiplier

# ❌ Avoid
@task
def process(value, multiplier=2):
    return value * multiplier
```

### 3. Inject Context Only When Needed

```python
# ✅ Simple computation - no context needed
@task
def add(x: int, y: int) -> int:
    return x + y

# ✅ Inter-task communication - needs context
@task(inject_context=True)
def share_data(ctx: TaskExecutionContext, value: int):
    ctx.get_channel().set("shared", value)
```

### 4. Use Descriptive Task IDs

```python
# ✅ Good - Clear and descriptive
fetch_user_profile = fetch(task_id="fetch_user_profile")
validate_email = validate(task_id="validate_email")

# ❌ Avoid - Generic names
task1 = fetch(task_id="t1")
task2 = validate(task_id="t2")
```

### 5. Get Results with ret_context

```python
# ✅ Good - Access all task results
_, ctx = wf.execute(ret_context=True)
result_a = ctx.get_result("task_a")
result_b = ctx.get_result("task_b")

# ⚠️ Limited - Only final result
result = wf.execute()  # Only last task's result
```

---

## Summary

### Learning Path

1. **Start Here**: [Level 1](#level-1-your-first-task) - Your first task
2. **Build Workflows**: [Level 2](#level-2-your-first-workflow) - Connect tasks
3. **Compose**: [Level 3](#level-3-task-composition) - Sequential and parallel
4. **Pass Data**: [Level 4](#level-4-passing-parameters) - Parameters and channels
5. **Reuse Code**: [Level 5](#level-5-task-instances) - Task instances
6. **Share State**: [Level 6](#level-6-channels-and-context) - Channels and context
7. **Control Execution**: [Level 7](#level-7-execution-patterns) - Execution patterns
8. **Complex Patterns**: [Level 8](#level-8-complex-workflows) - Diamond, multi-instance
9. **Advanced**: [Level 9](#level-9-dynamic-task-generation) - Dynamic tasks

### Parameter Priority

When resolving parameters, Graflow uses this order (highest priority wins):

```
Injection > Bound > Channel
   (ctx)    (task_id)  (initial_channel)
```

### Next Steps

**Explore Examples:**
- `examples/01_basics/` - Basic task patterns
- `examples/02_workflows/` - Workflow composition
- `examples/07_dynamic_tasks/` - Dynamic task generation

**Advanced Features:**
- [Checkpoint & Resume](checkpoint/checkpoint_resume_design.md) - Fault tolerance
- [HITL](hitl/hitl_design.md) - Human-in-the-loop workflows
- [Distributed Execution](scaling/redis_distributed_execution_redesign.md) - Scaling

**Core Files:**
- `graflow/core/task.py` - Task implementation
- `graflow/core/workflow.py` - Workflow context
- `graflow/core/engine.py` - Execution engine

---

**Made with ❤️ by the Graflow team**
