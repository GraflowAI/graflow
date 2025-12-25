# 03 - Data Flow and Inter-Task Communication

**Difficulty**: Intermediate
**Status**: ✅ Complete
**Prerequisites**: Complete [02_workflows](../02_workflows/) first

## Overview

This section demonstrates **data flow and inter-task communication** in Graflow - how tasks share data, state, and results through channels and the execution context.

## What You'll Learn

- 📡 Using channels for inter-task communication
- 🔒 Type-safe channels with TypedDict
- 💾 Storing and retrieving task results
- 🔄 Data flow patterns in workflows
- 📊 Sharing state across task boundaries

## Examples

### 1. channels_basic.py

**Concept**: Basic channel operations

Learn how to use channels for sharing data between tasks without direct parameter passing.

```bash
uv run python examples/03_data_flow/channels_basic.py
```

**Key Concepts**:
- Getting channel instances
- Setting and getting values
- Channel keys and existence checks
- Default values
- Channel lifecycle

---

### 2. typed_channels.py

**Concept**: Type-safe channels

Master TypedChannels for type-safe inter-task communication using TypedDict schemas.

```bash
uv run python examples/03_data_flow/typed_channels.py
```

**Key Concepts**:
- Defining message schemas with TypedDict
- Creating TypedChannel instances
- Type checking and validation
- IDE autocomplete support
- Schema evolution

---

### 3. results_storage.py

**Concept**: Task results and dependency data

Learn how to store task results and access them from dependent tasks.

```bash
uv run python examples/03_data_flow/results_storage.py
```

**Key Concepts**:
- Task result storage
- Retrieving results by task ID
- Building data pipelines
- Result propagation
- Error handling

---

## Data Flow Patterns

### Pattern 1: Shared Configuration

```python
@task(inject_context=True)
def setup(ctx: TaskExecutionContext):
    channel = ctx.get_channel()
    channel.set("config", {"batch_size": 100})

@task(inject_context=True)
def process(ctx: TaskExecutionContext):
    channel = ctx.get_channel()
    config = channel.get("config")
    batch_size = config["batch_size"]
```

### Pattern 2: Accumulating State

```python
@task(inject_context=True)
def process_batch(ctx: TaskExecutionContext):
    channel = ctx.get_channel()

    # Get current state
    total = channel.get("total_processed", 0)

    # Update state
    total += 100
    channel.set("total_processed", total)
```

### Pattern 3: Type-Safe Messages

```python
from typing import TypedDict

class MetricsMessage(TypedDict):
    count: int
    status: str
    timestamp: float

@task(inject_context=True)
def collect_metrics(ctx: TaskExecutionContext):
    typed_channel = ctx.get_typed_channel(MetricsMessage)

    metrics: MetricsMessage = {
        "count": 100,
        "status": "success",
        "timestamp": time.time()
    }
    typed_channel.set("metrics", metrics)
```

### Pattern 4: Result Dependencies

```python
with workflow("pipeline") as ctx:
    @task
    def fetch():
        return {"data": [1, 2, 3]}

    @task(inject_context=True)
    def process(ctx: TaskExecutionContext):
        # Get result from previous task
        data = ctx.get_result("fetch")
        return {"processed": len(data["data"])}

    fetch >> process
```

## Channel Types

Graflow supports different channel backends:

### MemoryChannel (Default)
- **Use**: Single-process workflows
- **Scope**: Process-local
- **Persistence**: No (in-memory only)

```python
context = ExecutionContext.create(
    graph, "start",
    channel_backend="memory"
)
```

### RedisChannel
- **Use**: Distributed workflows
- **Scope**: Across processes/workers
- **Persistence**: Redis server

```python
context = ExecutionContext.create(
    graph, "start",
    channel_backend="redis",
    config={"redis_client": redis_client}
)
```

## Best Practices

### ✅ DO

1. **Use channels for shared state**
   ```python
   channel.set("shared_config", config)
   ```

2. **Use TypedChannels for complex data**
   ```python
   typed_channel = ctx.get_typed_channel(MySchema)
   ```

3. **Provide default values**
   ```python
   value = channel.get("key", default=0)
   ```

4. **Use descriptive keys**
   ```python
   channel.set("user_auth_token", token)
   ```

### ❌ DON'T

1. **Don't store large objects**
   ```python
   # Bad: 100MB dataset
   channel.set("data", huge_dataset)

   # Good: Store reference or path
   channel.set("data_path", "/path/to/data")
   ```

2. **Don't use channels for task parameters**
   ```python
   # Bad: Using channel for direct parameters
   @task(inject_context=True)
   def process(ctx):
       x = ctx.get_channel().get("x")

   # Good: Use function parameters
   @task
   def process(x: int):
       return x * 2
   ```

3. **Don't assume key existence**
   ```python
   # Bad: May raise KeyError
   value = channel.get("key")

   # Good: Use default
   value = channel.get("key", default=None)
   ```

## Channel Operations

### Basic Operations

```python
channel = ctx.get_channel()

# Set value
channel.set("key", "value")

# Get value
value = channel.get("key")

# Get with default
value = channel.get("key", default="default")

# Check existence
if "key" in channel.keys():
    value = channel.get("key")

# List all keys
all_keys = channel.keys()
```

### TypedChannel Operations

```python
from typing import TypedDict

class MyMessage(TypedDict):
    count: int
    status: str

typed_channel = ctx.get_typed_channel(MyMessage)

# Set typed message
message: MyMessage = {"count": 42, "status": "ok"}
typed_channel.set("msg", message)

# Get typed message (with IDE support)
msg = typed_channel.get("msg")
print(msg["count"])  # IDE knows this is int
```

## Data Flow vs Parameters

### When to Use Channels
- ✅ Shared configuration across many tasks
- ✅ Accumulating state (counters, logs)
- ✅ Broadcasting data to multiple consumers
- ✅ Decoupling task implementations

### When to Use Parameters
- ✅ Direct task-to-task data passing
- ✅ Required inputs for task execution
- ✅ Simple, linear data flow
- ✅ Type-checked function arguments

## Common Use Cases

### Configuration Management
```python
@task(inject_context=True)
def load_config(ctx: TaskExecutionContext):
    config = {"db": "postgres", "batch": 100}
    ctx.get_channel().set("config", config)

@task(inject_context=True)
def use_config(ctx: TaskExecutionContext):
    config = ctx.get_channel().get("config")
    # All tasks can access config
```

### Metrics Collection
```python
@task(inject_context=True)
def track_metrics(ctx: TaskExecutionContext):
    channel = ctx.get_channel()
    metrics = channel.get("metrics", {})
    metrics["processed"] = metrics.get("processed", 0) + 1
    channel.set("metrics", metrics)
```

### Error Accumulation
```python
@task(inject_context=True)
def handle_error(ctx: TaskExecutionContext):
    channel = ctx.get_channel()
    errors = channel.get("errors", [])
    errors.append({"task": ctx.task_id, "error": "..."})
    channel.set("errors", errors)
```

### Resource Sharing
```python
@task(inject_context=True)
def init_resources(ctx: TaskExecutionContext):
    # Store connection string (not the connection itself!)
    ctx.get_channel().set("db_url", "postgresql://...")

@task(inject_context=True)
def use_resource(ctx: TaskExecutionContext):
    db_url = ctx.get_channel().get("db_url")
    # Create connection from URL
```

## Debugging Tips

1. **Inspect channel state**:
   ```python
   channel = ctx.get_channel()
   print(f"Channel keys: {channel.keys()}")
   for key in channel.keys():
       print(f"{key}: {channel.get(key)}")
   ```

2. **Log channel operations**:
   ```python
   channel.set("key", value)
   print(f"[{ctx.task_id}] Set channel key 'key' = {value}")
   ```

3. **Check for key existence**:
   ```python
   if "expected_key" not in channel.keys():
       print(f"Warning: Expected key not found in channel")
   ```

## Next Steps

After mastering data flow:

1. **04_execution**: Learn custom execution handlers
2. **05_distributed**: Scale workflows with distributed execution
3. **06_advanced**: Explore advanced patterns like cycles and dynamic tasks

## API Reference

**Channel**:
- `channel.set(key, value)` - Store value
- `channel.get(key, default=None)` - Retrieve value
- `channel.keys()` - List all keys

**TypedChannel**:
- `typed_channel = ctx.get_typed_channel(SchemaClass)` - Create typed channel
- `typed_channel.set(key, message)` - Store typed message
- `typed_channel.get(key)` - Retrieve typed message

**ExecutionContext**:
- `ctx.get_channel()` - Get channel instance
- `ctx.get_typed_channel(MessageType)` - Get typed channel
- `ctx.get_result(task_id, default=None)` - Get task result

---

**Ready to master data flow? Start with `channels_basic.py`! 🚀**
