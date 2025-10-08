# 04 - Custom Execution Handlers

**Difficulty**: Advanced
**Status**: ✅ Complete
**Prerequisites**: Complete [03_data_flow](../03_data_flow/) first

## Overview

This section demonstrates **custom execution handlers** in Graflow - how to control where and how tasks execute. Handlers allow you to run tasks in different environments:
- In-process (default)
- Docker containers
- Remote workers
- Custom execution environments

## What You'll Learn

- 🔧 Understanding task handlers
- 🐳 Executing tasks in Docker containers
- 🎨 Creating custom handlers
- ⚙️ Handler registration and selection
- 🔄 Serialization for remote execution

## Examples

### 1. direct_handler.py

**Concept**: Direct (in-process) execution

Understand the default DirectTaskHandler and when to use it.

```bash
python examples/04_execution/direct_handler.py
```

**Key Concepts**:
- Direct task execution
- Handler type specification
- Default handler behavior
- Performance characteristics

---

### 2. docker_handler.py

**Concept**: Docker container execution

Learn how to execute tasks in isolated Docker containers.

```bash
python examples/04_execution/docker_handler.py
```

**Key Concepts**:
- Docker handler usage
- Container configuration
- Image specification
- Environment isolation
- Result retrieval from containers

**Prerequisites**:
- Docker installed and running
- `pip install docker` (Docker SDK for Python)

---

### 3. custom_handler.py

**Concept**: Custom handler implementation

Build your own custom task handler for specialized execution needs.

```bash
python examples/04_execution/custom_handler.py
```

**Key Concepts**:
- TaskHandler interface
- Handler implementation
- Handler registration
- Custom execution logic
- Error handling

---

## Execution Handler Architecture

### Handler Interface

```python
from abc import ABC, abstractmethod
from graflow.core.handler import TaskHandler

class MyHandler(TaskHandler):
    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        # Execute task and store result
        result = task.run()
        context.set_result(task.task_id, result)
```

### Handler Selection

Tasks specify their handler using the `handler_type` parameter:

```python
@task(handler_type="direct")
def local_task():
    return "executed locally"

@task(handler_type="docker")
def container_task():
    return "executed in container"
```

### Handler Registration

Register custom handlers with the workflow engine:

```python
from graflow.core.engine import WorkflowEngine

engine = WorkflowEngine()
engine.register_handler("my_handler", MyCustomHandler())
```

## Built-in Handlers

### DirectTaskHandler (default)

- **Execution**: In-process, same Python interpreter
- **Isolation**: None (shares memory and state)
- **Performance**: Fastest (no overhead)
- **Use case**: Most tasks, local development

```python
@task(handler_type="direct")  # or omit (default)
def my_task():
    return "result"
```

### DockerTaskHandler

- **Execution**: Docker container
- **Isolation**: Full process and filesystem isolation
- **Performance**: Higher overhead (container startup)
- **Use case**: Untrusted code, environment dependencies, reproducibility

```python
@task(handler_type="docker", docker_image="python:3.11-slim")
def isolated_task():
    return "result from container"
```

## When to Use Each Handler

### Use DirectTaskHandler (default)

✅ Trusted code
✅ Local development
✅ Fast iteration
✅ Shared dependencies
✅ Low latency requirements

### Use DockerTaskHandler

✅ Untrusted or experimental code
✅ Different Python versions
✅ Specific system dependencies
✅ Reproducible environments
✅ Security isolation

### Use Custom Handler

✅ Remote execution (SSH, cloud, etc.)
✅ Specialized hardware (GPU, TPU)
✅ External systems (APIs, databases)
✅ Custom execution logic
✅ Monitoring and telemetry

## Handler Patterns

### Pattern 1: Mixed Handlers in One Workflow

```python
with workflow("mixed") as ctx:
    @task(handler_type="direct")
    def fast_local():
        return "local"

    @task(handler_type="docker", docker_image="python:3.11")
    def isolated():
        return "container"

    fast_local >> isolated
```

### Pattern 2: Conditional Handler Selection

```python
import os

handler = "docker" if os.getenv("PRODUCTION") else "direct"

@task(handler_type=handler)
def adaptive_task():
    return "result"
```

### Pattern 3: Handler with Configuration

```python
@task(
    handler_type="docker",
    docker_image="python:3.11-slim",
    docker_volumes={"/host/path": "/container/path"},
    docker_environment={"API_KEY": "secret"}
)
def configured_task():
    return "result"
```

## Serialization Considerations

### What Gets Serialized

When using remote handlers (Docker, distributed), Graflow serializes:
- ✅ Task function and code
- ✅ ExecutionContext state
- ✅ Channel data
- ✅ Task parameters

### Serialization Requirements

- Use cloudpickle for lambda and closure support
- Avoid unpicklable objects (threads, locks, connections)
- Store connection strings, not connections
- Use primitive types when possible

### Example: Serialization-Safe Task

```python
@task(handler_type="docker")
def safe_task(data: dict) -> dict:
    # Good: primitive types
    result = {"count": len(data["items"])}
    return result
```

### Example: Serialization Issues

```python
# BAD: Don't do this
import threading

@task(handler_type="docker")
def unsafe_task():
    lock = threading.Lock()  # Cannot serialize!
    return lock
```

## Best Practices

### ✅ DO

1. **Use appropriate handlers**
   - Default to DirectTaskHandler
   - Use Docker for isolation needs
   - Create custom handlers for specialized execution

2. **Handle errors properly**
   ```python
   def execute_task(self, task, context):
       try:
           result = task.run()
           context.set_result(task.task_id, result)
       except Exception as e:
           context.set_result(task.task_id, e)
   ```

3. **Document handler requirements**
   ```python
   @task(
       handler_type="docker",
       docker_image="python:3.11-slim"
   )
   def my_task():
       """Execute in Docker container.

       Requires: Docker installed and running
       Image: python:3.11-slim
       """
       pass
   ```

### ❌ DON'T

1. **Don't use remote handlers unnecessarily**
   ```python
   # BAD: Unnecessary overhead
   @task(handler_type="docker")
   def simple_math(a, b):
       return a + b
   ```

2. **Don't forget error handling**
   ```python
   # BAD: No error handling
   def execute_task(self, task, context):
       result = task.run()  # What if this fails?
       context.set_result(task.task_id, result)
   ```

3. **Don't mix execution environments carelessly**
   ```python
   # BAD: State won't transfer properly
   shared_state = []

   @task(handler_type="direct")
   def task1():
       shared_state.append(1)  # Modifies in-process state

   @task(handler_type="docker")
   def task2():
       # Can't see shared_state! Different process!
       return len(shared_state)
   ```

## Performance Comparison

| Handler | Startup | Execution | Isolation | Use Case |
|---------|---------|-----------|-----------|----------|
| Direct | ~0ms | Native | None | Most tasks |
| Docker | ~500-2000ms | Near-native | Full | Isolation needed |
| Custom | Varies | Varies | Configurable | Specialized |

## Debugging Handlers

### Enable Handler Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("graflow.handlers")
```

### Test Handler Execution

```python
# Test in isolation
handler = MyCustomHandler()
task = Task("test_task")
context = ExecutionContext.create(graph, "test")

handler.execute_task(task, context)
result = context.get_result("test_task")
print(f"Result: {result}")
```

### Common Issues

1. **Docker not running**
   ```
   Error: Cannot connect to Docker daemon
   Solution: Start Docker Desktop or Docker service
   ```

2. **Serialization failures**
   ```
   Error: cannot pickle 'thread.lock' object
   Solution: Remove unpicklable objects from task
   ```

3. **Handler not found**
   ```
   Error: Unknown handler type: my_handler
   Solution: Register handler with engine.register_handler()
   ```

## Next Steps

After mastering execution handlers:

1. **05_distributed**: Learn distributed execution with Redis workers
2. **06_advanced**: Explore advanced patterns
3. **07_real_world**: Apply concepts to real-world problems

## API Reference

**TaskHandler**:
- `execute_task(task, context)` - Execute task and store result

**Task Decorator**:
- `@task(handler_type="direct")` - Specify handler type
- `@task(handler_type="docker", docker_image="...")` - Docker configuration

**WorkflowEngine**:
- `engine.register_handler(type, handler)` - Register custom handler

---

**Ready to master execution handlers? Start with `direct_handler.py`! 🚀**
