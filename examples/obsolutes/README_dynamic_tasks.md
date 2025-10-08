# Dynamic Task Generation Examples

This directory contains examples demonstrating Graflow's dynamic task generation capabilities using `context.next_task()` and `context.next_iteration()` methods.

## Overview

Dynamic Task Generation allows workflows to create new tasks at runtime based on execution results, enabling:

- **Conditional Processing**: Create different tasks based on data conditions
- **Iterative Workflows**: Generate repeated tasks with updated data  
- **Adaptive Execution**: Respond to runtime conditions with appropriate tasks
- **Error Recovery**: Create retry or error handling tasks dynamically
- **Goto Jumps**: Jump directly to existing tasks, bypassing normal successor execution

## Key Methods

### `context.next_task(executable: Executable, goto: bool = False) -> str`

Creates a new task dynamically during execution or jumps to existing task.

```python
# Create new dynamic task (normal successor processing)
task = TaskWrapper("processor", lambda: process_data())
task_id = context.next_task(task)

# Jump to existing task (skip current task's successors)
existing_task = graph.nodes["validator"]["task"]
task_id = context.next_task(existing_task, goto=True)
```

**Parameters:**
- `executable`: TaskWrapper or other Executable object to execute
- `goto`: If `True`, jump to existing task and skip current task's successors

### `context.next_iteration(data: Any = None, task_id: Optional[str] = None) -> str`

Creates an iteration of the current task with new data (for cycles).

```python
# Continue iteration with updated data
context.next_iteration({"count": count + 1, "limit": 5})
```

## Examples

### 1. `dynamic_tasks_simple.py` - Core Functionality

**Best starting point** - Demonstrates the basic mechanics:

- ✅ Basic `next_task()` usage
- ✅ Basic `next_iteration()` usage  
- ✅ Conditional task creation
- ✅ Iteration chains

```bash
python examples/dynamic_tasks_simple.py
```

### 2. `dynamic_task_demo.py` - Workflow Integration

Shows integration with workflow context (work in progress):

- 🔄 Integration with `@task` decorator
- 🔄 Workflow context integration
- 🔄 Complex scenarios

### 3. `dynamic_task_generation.py` - Advanced Examples

Comprehensive examples covering:

- 🔄 Error handling with recovery
- 🔄 Batch processing
- 🔄 Human-in-the-loop workflows
- 🔄 Multi-condition branching

## Usage Patterns

### Pattern 1: Conditional Task Creation

```python
@task(id="processor")
def process_data(context, data):
    value = data.get("value", 0)
    
    if value > 100:
        high_task = TaskWrapper("high_processor", lambda: handle_high_value(value))
        context.next_task(high_task)
    else:
        low_task = TaskWrapper("low_processor", lambda: handle_low_value(value))
        context.next_task(low_task)
    
    return {"processed": True}
```

### Pattern 2: Iterative Processing

```python
@task(id="optimizer")
def optimize(context, params):
    # Update parameters
    new_params = update_parameters(params)
    
    if not converged(new_params):
        # Continue optimization
        context.next_iteration(new_params)
    else:
        # Create completion task
        done_task = TaskWrapper("save_results", lambda: save_model(new_params))
        context.next_task(done_task)
    
    return new_params
```

### Pattern 3: Error Recovery

```python
@task(id="resilient_task")
def process_with_retry(context, data):
    retry_count = data.get("retry_count", 0)
    
    try:
        result = risky_operation(data)
        # Success - create success handler
        success_task = TaskWrapper("handle_success", lambda: log_success(result))
        context.next_task(success_task)
        
    except Exception as e:
        if retry_count < 3:
            # Retry
            retry_data = data.copy()
            retry_data["retry_count"] = retry_count + 1
            context.next_iteration(retry_data)
        else:
            # Give up - create error handler
            error_task = TaskWrapper("handle_error", lambda: log_error(e))
            context.next_task(error_task)
```

### Pattern 4: Goto Jump to Existing Tasks

Jump to existing task nodes in the graph, skipping normal successor execution:

```python
@task(id="controller", inject_context=True)
def controller_task(task_ctx):
    condition = analyze_condition()
    
    if condition == "emergency":
        # Jump directly to emergency handler (skip normal successors)
        emergency_task = graph.nodes["emergency_handler"]["task"]
        task_ctx.next_task(emergency_task, goto=True)
        
    elif condition == "validation_needed":
        # Jump to validation task (skip normal successors)
        validator_task = graph.nodes["data_validator"]["task"]
        task_ctx.next_task(validator_task, goto=True)
        
    else:
        # Normal processing - create new dynamic task (successors will run)
        processor = TaskWrapper("normal_processor", lambda: process_normal())
        task_ctx.next_task(processor)  # goto=False (default)
    
    return {"condition": condition}

# In graph setup:
# controller_task -> normal_successor  # This edge is SKIPPED when goto=True
# emergency_handler and data_validator are existing nodes in graph
```

#### Key Differences:

- **`next_task(task, goto=True)`**: Jump to existing task, **skip current task's successors**
- **`next_task(task)`** or **`next_task(task, goto=False)`**: Create/execute task, **normal successor processing**

```python
# Example with mixed usage:
@task(id="dispatcher", inject_context=True)  
def dispatcher(task_ctx):
    # Create new dynamic task (successors will run normally)
    worker = TaskWrapper("worker", work_function)
    task_ctx.next_task(worker)
    
    # Jump to existing special task (skip dispatcher's successors)
    special = graph.nodes["special_handler"]["task"]
    task_ctx.next_task(special, goto=True)
    
    return "dispatched"

# Result: worker -> special_handler (dispatcher's successors are skipped)
```

## Key Concepts

### TaskWrapper Creation

Always create `TaskWrapper` objects for dynamic tasks:

```python
# ✅ Correct
task = TaskWrapper("task_id", function)
context.next_task(task)

# ❌ Incorrect - next_task only accepts Executable objects
context.next_task(function)  # TypeError!
```

### Task ID Management

`TaskWrapper` automatically manages task IDs:

```python
task = TaskWrapper("my_task", function)
print(task.task_id)  # "my_task"

# context.next_task() uses the task's ID automatically
task_id = context.next_task(task)  # Returns "my_task"
```

### Execution Context

Dynamic tasks need access to `ExecutionContext`:

```python
@task(id="my_task")
def my_task(context, data):  # context is ExecutionContext
    # Use context.next_task() and context.next_iteration()
    pass
```

## Implementation Notes

### Simplicity Design

The implementation prioritizes simplicity:

- `next_task()` only accepts `Executable` objects
- Task ID management is handled by the `Executable`
- No complex type checking or conversion logic

### Responsibility Separation

- **ExecutionContext**: Manages graph and queue operations
- **TaskWrapper**: Handles task ID and function wrapping
- **User Code**: Creates appropriate `TaskWrapper` objects

### Integration with Existing Features

Dynamic tasks work with all existing Graflow features:

- Cycle detection and limits
- Channel-based communication
- Error handling and retries
- Workflow context management

### Goto Jump Control

The `goto` parameter provides fine-grained control over workflow execution:

- **`goto=False` (default)**: Normal execution, successors of current task will run
- **`goto=True`**: Jump behavior, successors of current task are skipped
- **Auto-detection**: When `goto=False`, automatically detects existing vs new tasks
- **Explicit control**: Use `goto=True` for conditional workflow routing

## Running the Examples

```bash
# Run the simple example (recommended first)
python examples/dynamic_tasks_simple.py

# Run other examples
python examples/dynamic_task_demo.py
python examples/dynamic_task_generation.py
```

## Next Steps

1. Start with `dynamic_tasks_simple.py` to understand the basics
2. Experiment with conditional task creation
3. Try iterative processing patterns
4. Integrate with your own workflow scenarios

The dynamic task generation feature enables powerful, adaptive workflows that can respond to runtime conditions and data patterns dynamically.