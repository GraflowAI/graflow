# Queue Task Creation Fix - Fundamental Solution

## Problem Statement

`InMemoryTaskQueue.__init__()` creates NEW `Task` objects instead of using existing tasks from the graph. This violates the single source of truth principle and creates unnecessary coupling.

## Current Implementation (WRONG)

**File**: `graflow/queue/memory.py:15-27`

```python
class InMemoryTaskQueue(TaskQueue):
    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue: deque[TaskSpec] = deque()
        if start_node:
            # ❌ WRONG: Creates a brand new Task object
            from graflow.core.task import Task
            start_task = Task(start_node, register_to_context=False)

            task_spec = TaskSpec(
                executable=start_task,  # ❌ Different object than graph's task
                execution_context=execution_context
            )
            self._queue.append(task_spec)
            self._task_specs[start_node] = task_spec
```

### Problems

1. **Duplicate Task Objects**:
   - Graph has: `TaskWrapper("my_task", func)` with full metadata
   - Queue has: `Task("my_task")` without metadata
   - Two different objects representing the same logical task

2. **Lost Information**:
   - Original task's function, handler_type, context injection settings are lost
   - Only task_id is preserved

3. **Serialization Dependency**:
   - Requires Task class to have serialization attributes
   - Wouldn't be needed if queue used graph's tasks

4. **Execution Mismatch**:
   - Queue's Task is a placeholder
   - Actual execution uses graph's TaskWrapper
   - Confusing two-phase resolution

## Correct Implementation (OPTION 1 - Recommended)

**Use Task from Graph**

```python
class InMemoryTaskQueue(TaskQueue):
    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue: deque[TaskSpec] = deque()
        if start_node:
            # ✅ CORRECT: Get task from graph (single source of truth)
            start_task = execution_context.graph.get_node(start_node)

            if start_task is None:
                raise ValueError(
                    f"Start node '{start_node}' not found in graph. "
                    f"Available nodes: {list(execution_context.graph.nodes.keys())}"
                )

            task_spec = TaskSpec(
                executable=start_task,  # ✅ Same object as in graph
                execution_context=execution_context
            )
            self._queue.append(task_spec)
            self._task_specs[start_node] = task_spec
```

### Benefits

✅ **Single Source of Truth**: Graph owns all tasks
✅ **No Duplication**: Queue references graph's tasks
✅ **Consistent State**: Same task object everywhere
✅ **Rich Information**: Full task metadata available
✅ **Type Safety**: Ensures start_node exists before execution
✅ **Clear Semantics**: Queue holds references, not copies

### Requirements

- Graph must be populated before ExecutionContext creation
- Start node must exist in graph (validated with helpful error)

## Correct Implementation (OPTION 2 - Lazy Initialization)

**Defer Task Resolution**

```python
class InMemoryTaskQueue(TaskQueue):
    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue: deque[TaskSpec] = deque()
        self._pending_start_node: Optional[str] = start_node
        # Don't enqueue yet - wait for first access

    def _ensure_start_node_enqueued(self) -> None:
        """Enqueue start node on first access (lazy)."""
        if self._pending_start_node is not None:
            start_task = self.execution_context.graph.get_node(self._pending_start_node)

            if start_task is None:
                raise ValueError(
                    f"Start node '{self._pending_start_node}' not found in graph. "
                    f"Available nodes: {list(self.execution_context.graph.nodes.keys())}"
                )

            task_spec = TaskSpec(
                executable=start_task,
                execution_context=self.execution_context
            )
            self._queue.append(task_spec)
            self._task_specs[self._pending_start_node] = task_spec
            self._pending_start_node = None  # Mark as enqueued

    def dequeue(self) -> Optional[TaskSpec]:
        """Get next TaskSpec (triggers lazy initialization)."""
        self._ensure_start_node_enqueued()

        if self._queue:
            task_spec = self._queue.popleft()
            task_spec.status = TaskStatus.RUNNING
            if self.enable_metrics:
                self.metrics['dequeued'] += 1
            return task_spec
        return None

    def get_next_task(self) -> Optional[str]:
        """Get next task ID (legacy compatibility)."""
        self._ensure_start_node_enqueued()
        task_spec = self.dequeue()
        return task_spec.task_id if task_spec else None
```

### Benefits

✅ **Flexible Initialization**: Graph can be populated after context creation
✅ **Single Source of Truth**: Still uses graph's tasks
✅ **Backward Compatible**: Doesn't change external API

### Drawbacks

⚠️ **More Complex**: Additional state and logic
⚠️ **Hidden Behavior**: Initialization happens on first access
⚠️ **Error Timing**: Errors delayed until first dequeue

## Recommendation

**Use Option 1 (Direct Graph Access)** because:

1. **Simpler**: Less code, less state, easier to understand
2. **Fail Fast**: Errors at context creation time (better DX)
3. **Clear Contract**: Graph must be complete before context creation
4. **Explicit**: No hidden lazy initialization

## Migration Path

### Step 1: Update InMemoryTaskQueue

```python
# graflow/queue/memory.py
class InMemoryTaskQueue(TaskQueue):
    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue: deque[TaskSpec] = deque()
        if start_node:
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

### Step 2: Update RedisTaskQueue (if similar issue exists)

Check if RedisTaskQueue has the same problem and apply similar fix.

### Step 3: Run Tests

```bash
PYTHONPATH=. uv run pytest tests/core/test_checkpoint.py -v
PYTHONPATH=. uv run pytest tests/queue/ -v
```

All tests should still pass since:
- Task and TaskWrapper now have serialization attributes
- Queue now uses graph's tasks (proper design)

### Step 4: Optional Cleanup

If desired, serialization attributes could be removed from Task class since queue no longer creates Task objects. However, keeping them is harmless and provides defense in depth.

## Design Principles

This fix enforces key design principles:

1. **Single Source of Truth**: Graph is the definitive task registry
2. **Reference, Don't Copy**: Queue holds references to graph's tasks
3. **Validate Early**: Check start_node exists before execution starts
4. **Clear Ownership**: Graph owns tasks, queue manages execution order
5. **Type Consistency**: Same task object type throughout system

## Impact Analysis

### Backward Compatibility

✅ **Compatible**: No API changes to ExecutionContext or TaskQueue
✅ **Better Errors**: Now validates start_node exists (improvement)
⚠️ **Requirement Change**: Graph must be populated before context creation

### Performance

✅ **Better**: No object creation overhead
✅ **Less Memory**: No duplicate task objects
✅ **Faster**: Direct graph access vs. Task instantiation

### Code Quality

✅ **Simpler**: Fewer lines of code
✅ **Clearer**: Obvious that queue references graph
✅ **Maintainable**: One less place to update when Task class changes

## Test Changes

Tests should be **updated** to ensure graph is populated:

```python
# BEFORE (worked by accident due to our fix)
context = ExecutionContext.create(graph, start_node="my_task")

# AFTER (explicit and correct)
graph = TaskGraph()
my_task = create_task()  # @task decorator or TaskWrapper
graph.add_node(my_task, "my_task")

# Now context creation validates that my_task exists
context = ExecutionContext.create(graph, start_node="my_task")
```

This makes tests more explicit about dependencies.

## Related Issues

### WorkflowEngine

Check if WorkflowEngine or other components have similar issues where they create tasks instead of using graph's tasks.

### Task Creation

Ensure all task creation goes through:
1. `@task` decorator → adds to graph
2. `graph.add_node(task, task_id)` → explicit graph population
3. Context/queue use → references from graph

Never: Create ad-hoc Task objects for execution

## Conclusion

The **fundamental fix** is to make `InMemoryTaskQueue` use tasks from the graph instead of creating new ones. This:

- Fixes the root cause
- Simplifies the codebase
- Enforces correct design patterns
- Provides better error messages
- Reduces memory usage

The serialization attribute fix we applied is still valuable as **defense in depth**, but the proper solution is to eliminate the need for it by using graph's tasks directly.
