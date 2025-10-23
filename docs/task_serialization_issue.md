# Task Serialization Issue: Deep Dive

## Issue Summary

When `ExecutionContext.create()` is called with a `start_node` parameter, the `InMemoryTaskQueue` creates a plain `Task` object that cannot be serialized during checkpoint creation, causing `AttributeError: 'Task' object has no attribute '__name__'`.

## Problem Flow

### 1. Context Creation with start_node

```python
# User code
graph = TaskGraph()

@task("my_task")
def my_task_func():
    return "result"

graph.add_node(my_task_func, "my_task")

# This triggers the issue
context = ExecutionContext.create(graph, start_node="my_task", max_steps=10)
```

### 2. InMemoryTaskQueue Initialization

**Location**: `graflow/queue/memory.py:15-27`

```python
class InMemoryTaskQueue(TaskQueue):
    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue: deque[TaskSpec] = deque()
        if start_node:
            # ❌ PROBLEM: Creates a NEW Task object instead of using graph's task
            from graflow.core.task import Task
            start_task = Task(start_node, register_to_context=False)

            task_spec = TaskSpec(
                executable=start_task,
                execution_context=execution_context
            )
            self._queue.append(task_spec)
            self._task_specs[start_node] = task_spec
```

**The Issue**:
- Creates a **new** `Task("my_task")` object
- This is **different** from the `@task` decorated function in the graph
- The new `Task` object is a **bare instance** without serialization attributes

### 3. Task Objects Comparison

#### Graph's Task (from @task decorator)
```python
# graflow/core/decorators.py:26-96
@task("my_task")
def my_task_func():
    return "result"

# This creates a TaskWrapper with:
task_obj = TaskWrapper(task_id, wrapper, inject_context=False, handler_type=None)

# Decorator adds serialization attributes (lines 76-86):
task_obj.__name__ = f.__name__              # ✅ "my_task_func"
task_obj.__module__ = f.__module__          # ✅ "__main__" or module path
task_obj.__qualname__ = f.__qualname__      # ✅ "my_task_func"
task_obj.__doc__ = f.__doc__                # ✅ Docstring
task_obj.__annotations__ = f.__annotations__ # ✅ Type hints
```

**Result**: TaskWrapper has all required attributes for serialization

#### Queue's Task (created by InMemoryTaskQueue)
```python
# graflow/core/task.py
start_task = Task(start_node, register_to_context=False)

# Task class definition (simplified):
class Task(Executable):
    def __init__(self, task_id: str, register_to_context: bool = True):
        self.task_id = task_id
        self.execution_context = None
        # ❌ NO __name__ attribute
        # ❌ NO __module__ attribute
        # ❌ NO __qualname__ attribute
        # ❌ NO __doc__ attribute
```

**Result**: Plain Task object **missing** serialization attributes

### 4. Checkpoint Creation Triggers Serialization

**Location**: `graflow/core/checkpoint.py:78-104`

```python
def create_checkpoint(cls, context, path=None, metadata=None):
    # ...

    # Step 3: Get pending tasks from queue
    pending_specs = cls._get_pending_task_specs(context)  # Returns [TaskSpec]

    # Step 4: Serialize TaskSpec data for JSON persistence
    serialized_specs = [cls._serialize_task_spec(spec) for spec in pending_specs]
    # ❌ This is where serialization fails
```

### 5. TaskSpec Serialization

**Location**: `graflow/core/checkpoint.py:233-243`

```python
@staticmethod
def _serialize_task_spec(task_spec: TaskSpec) -> Dict[str, Any]:
    return {
        "task_id": task_spec.task_id,
        "task_data": task_spec.task_data,  # ❌ This triggers the error
        "status": task_spec.status.value,
        "created_at": task_spec.created_at,
        "strategy": task_spec.strategy,
        "retry_count": task_spec.retry_count,
        "max_retries": task_spec.max_retries,
        "last_error": task_spec.last_error,
    }
```

### 6. task_data Property Triggers Serialization

**Location**: `graflow/queue/base.py:45-49`

```python
@property
def task_data(self) -> Any:
    """Serialized task data (lazy)."""
    return self.execution_context.task_resolver.serialize_task(
        self.executable,  # ❌ This is the bare Task object
        self.strategy
    )
```

### 7. Task Serialization Attempts Reference Strategy

**Location**: `graflow/core/task_registry.py:88-92`

```python
def serialize_task(self, task: Executable, strategy: str = "reference") -> Dict[str, Any]:
    """Serialize a task using the specified strategy."""
    if strategy == "reference":
        return TaskSerializer._serialize_reference(task)  # ❌ Fails here
    elif strategy == "pickle":
        return TaskSerializer._serialize_pickle(task)
```

### 8. The Fatal Error

**Location**: `graflow/core/task_registry.py:102-106`

```python
@staticmethod
def _serialize_reference(task: Executable) -> Dict[str, Any]:
    """Serialize task as importable reference."""
    return {
        "strategy": "reference",
        "module": task.__module__,
        "name": task.__name__,        # ❌ AttributeError: 'Task' object has no attribute '__name__'
        "qualname": getattr(task, "__qualname__", task.__name__)
    }
```

**Error Message**:
```
AttributeError: 'Task' object has no attribute '__name__'
  File "graflow/core/task_registry.py", line 105, in _serialize_reference
    "name": task.__name__,
            ^^^^^^^^^^^^^
```

## Why This Happens

### The Mismatch

1. **Graph contains**: `TaskWrapper` (from `@task` decorator) with full serialization support
2. **Queue contains**: `Task` (created by `InMemoryTaskQueue`) without serialization support

### The Root Cause

`InMemoryTaskQueue` doesn't **reuse** the task from the graph. Instead, it **creates a new** plain `Task` object:

```python
# Should be:
start_task = execution_context.graph.get_node(start_node)  # ✅ Get from graph

# But currently is:
from graflow.core.task import Task
start_task = Task(start_node, register_to_context=False)   # ❌ Create new
```

## Why start_node=None Works

When `start_node=None`:

```python
context = ExecutionContext.create(graph, start_node=None, max_steps=10)
```

1. **No automatic task creation** - Queue starts empty
2. **Tasks added later** come from the graph (via `next_task()` or manual enqueue)
3. **Graph tasks have serialization** support (TaskWrapper from `@task` decorator)
4. **Checkpoint succeeds** because all tasks in queue are serializable

## Impact Scope

### Affected Code Paths

1. **Any checkpoint creation** when queue has tasks:
   - `CheckpointManager.create_checkpoint(context, ...)`
   - Tests that create checkpoints with pending tasks

2. **Any context with start_node**:
   - `ExecutionContext.create(graph, start_node="task", ...)`
   - Workflow execution with initial task

3. **Distributed execution**:
   - Worker startup with predefined start task
   - Task queue restoration from checkpoint

### Not Affected

1. **Contexts without start_node**: `ExecutionContext.create(graph, start_node=None, ...)`
2. **Empty queues**: Checkpoints before any tasks are enqueued
3. **Redis queue**: If tasks are serialized differently (TBD)

## Solutions

### Solution 1: Use Graph Task (Recommended)

**Location**: `graflow/queue/memory.py:15-27`

```python
class InMemoryTaskQueue(TaskQueue):
    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue: deque[TaskSpec] = deque()
        if start_node:
            # ✅ FIX: Get task from graph instead of creating new one
            start_task = execution_context.graph.get_node(start_node)
            if start_task is None:
                raise ValueError(f"Start node '{start_node}' not found in graph")

            task_spec = TaskSpec(
                executable=start_task,  # ✅ Use graph's TaskWrapper
                execution_context=execution_context
            )
            self._queue.append(task_spec)
            self._task_specs[start_node] = task_spec
```

**Pros**:
- ✅ Fixes root cause
- ✅ Ensures queue uses same tasks as graph
- ✅ No changes to Task class needed
- ✅ Consistent with design principle (single source of truth)

**Cons**:
- ⚠️ Requires task to exist in graph before queue initialization
- ⚠️ Raises error if start_node not in graph (but this is good validation)

### Solution 2: Add Serialization to Task Class

**Location**: `graflow/core/task.py`

```python
class Task(Executable):
    def __init__(self, task_id: str, register_to_context: bool = True):
        self.task_id = task_id
        self.execution_context = None

        # ✅ FIX: Add serialization attributes
        self.__name__ = task_id
        self.__module__ = self.__class__.__module__
        self.__qualname__ = f"{self.__class__.__name__}.{task_id}"
        self.__doc__ = f"Task: {task_id}"
```

**Pros**:
- ✅ Makes Task fully serializable
- ✅ No changes to queue initialization
- ✅ Backwards compatible

**Cons**:
- ⚠️ Doesn't address the design issue (duplicate tasks)
- ⚠️ Task and TaskWrapper still have different serialization paths
- ⚠️ May need to handle deserialization differently

### Solution 3: Use Pickle Strategy

**Location**: `graflow/queue/memory.py:15-27`

```python
class InMemoryTaskQueue(TaskQueue):
    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue: deque[TaskSpec] = deque()
        if start_node:
            from graflow.core.task import Task
            start_task = Task(start_node, register_to_context=False)

            task_spec = TaskSpec(
                executable=start_task,
                execution_context=execution_context,
                strategy="pickle"  # ✅ FIX: Use pickle instead of reference
            )
            self._queue.append(task_spec)
            self._task_specs[start_node] = task_spec
```

**Pros**:
- ✅ Works with any Executable (no __name__ required)
- ✅ Minimal code changes

**Cons**:
- ⚠️ Pickle strategy may not work for all task types
- ⚠️ Doesn't address the design issue (duplicate tasks)
- ⚠️ Inconsistent serialization strategy (some tasks by reference, some by pickle)

### Solution 4: Lazy Queue Initialization

**Location**: `graflow/queue/memory.py`

```python
class InMemoryTaskQueue(TaskQueue):
    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue: deque[TaskSpec] = deque()
        self._pending_start_node = start_node  # ✅ Store for later
        # Don't enqueue yet

    def get_next_task(self) -> Optional[str]:
        # ✅ FIX: Enqueue start_node on first access
        if self._pending_start_node:
            start_task = self.execution_context.graph.get_node(self._pending_start_node)
            if start_task:
                task_spec = TaskSpec(
                    executable=start_task,
                    execution_context=self.execution_context
                )
                self._queue.append(task_spec)
                self._task_specs[self._pending_start_node] = task_spec
            self._pending_start_node = None

        return super().get_next_task()
```

**Pros**:
- ✅ Defers task creation until graph is fully populated
- ✅ More flexible initialization order

**Cons**:
- ⚠️ More complex logic
- ⚠️ May affect queue state queries before first access
- ⚠️ Harder to debug

## Recommendation

**Use Solution 1: Use Graph Task**

This is the best solution because:

1. **Fixes root cause**: Tasks should come from graph, not be duplicated
2. **Single source of truth**: Graph is the definitive task registry
3. **Consistent behavior**: All tasks in queue are from graph
4. **Type safety**: Ensures start_node exists before execution
5. **Minimal changes**: Single location fix

## Testing Impact

### Before Fix: 15/31 tests passing

Failing tests all have this pattern:
```python
# ❌ Fails with AttributeError
context = ExecutionContext.create(graph, start_node="task_a", max_steps=10)
CheckpointManager.create_checkpoint(context, path=checkpoint_path)
```

### After Fix: 31/31 tests should pass

All tests will work because:
```python
# ✅ Queue uses TaskWrapper from graph
context = ExecutionContext.create(graph, start_node="task_a", max_steps=10)
# start_task is now graph.get_node("task_a") which has __name__, __module__, etc.
CheckpointManager.create_checkpoint(context, path=checkpoint_path)  # ✅ Success
```

## Code Changes Required

### File: `graflow/queue/memory.py`

```python
class InMemoryTaskQueue(TaskQueue):
    """In-memory task queue with TaskSpec support (Phase 1 implementation)."""

    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue: deque[TaskSpec] = deque()
        if start_node:
            # Get task from graph instead of creating new Task object
            start_task = execution_context.graph.get_node(start_node)
            if start_task is None:
                raise ValueError(
                    f"Start node '{start_node}' not found in graph. "
                    f"Available nodes: {list(execution_context.graph.nodes.keys())}"
                )
            task_spec = TaskSpec(
                executable=start_task,
                execution_context=execution_context
            )
            self._queue.append(task_spec)
            self._task_specs[start_node] = task_spec
```

**Lines changed**: 4 lines
- Line 20-21: Remove Task import and creation
- Line 20: Add task retrieval from graph
- Line 21-24: Add validation with helpful error message

## Verification

After applying the fix:

```bash
# Should show 31/31 passing
PYTHONPATH=. uv run pytest tests/core/test_checkpoint.py -v

# Should also work for scenario tests
PYTHONPATH=. uv run pytest tests/scenario/test_checkpoint_scenarios.py -v
```

## Related Issues

This issue is related to:
1. **Task lifecycle**: Who owns task instances?
2. **Serialization strategy**: Reference vs pickle
3. **Queue initialization**: When should tasks be enqueued?
4. **Graph completeness**: Should graph be complete before context creation?

## Design Implications

### Current Design
- Graph stores tasks
- Queue stores TaskSpec referencing tasks
- **Problem**: Queue creates new tasks instead of referencing graph's tasks

### Improved Design
- Graph is **single source of truth** for tasks
- Queue **references** tasks from graph
- TaskSpec holds **reference** to graph's task, not a copy
- Serialization uses task from graph (which has proper attributes)

This aligns with the checkpoint design document's principle: "Leverage existing serialization infrastructure."
