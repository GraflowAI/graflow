# Runtime Graph Tracing Hierarchy Fix Plan

## Problem Analysis

The user reported that `_runtime_graph` in `graflow/trace/base.py` fails to correctly capture the parent-child relationship (calling hierarchy) of tasks. Specifically, `parent_task_id` in `Tracer.on_task_start` often appears incorrect or missing.

### Root Cause
1.  **Dependency on Local Stack:** `Tracer.on_task_start` determines the `parent_task_id` exclusively by checking `context.current_task_id`.
    ```python
    # graflow/trace/base.py
    parent_task_id = None
    if hasattr(context, 'current_task_id') and context.current_task_id:
        parent_task_id = context.current_task_id
    ```
2.  **Stack State Timing:** `context.current_task_id` reflects the top of `self._task_execution_stack`.
    In `ExecutionContext.executing_task(task)`, the tracer hook `on_task_start` is called *before* the new task context is pushed to the stack.
    - **Local Nested Execution:** This works correctly. If Task A is running (on stack), and spawns Task B, `on_task_start(B)` sees A on the stack. `Parent = A`.
    - **Distributed/Worker Execution:** This fails. When a Worker processing a task (e.g., in a `ParallelGroup`) starts execution, it initializes a fresh `ExecutionContext` with an empty stack. `current_task_id` is `None`. The tracer sees no parent, resulting in a disconnected node in the runtime graph, even though `TaskSpec` carried the `parent_span_id`.

3.  **Missing Link:** Although `Worker` calls `tracer.attach_to_trace(..., parent_span_id=...)`, the base `Tracer` class does not store this `parent_span_id` in a way that `on_task_start` can access it as a fallback.

## Proposed Solution

Modify `graflow/trace/base.py` to make the `Tracer` aware of the external/distributed context parent.

### 1. Update `Tracer` Class
Add an attribute to store the external parent span ID.

```python
class Tracer(ABC):
    def __init__(self, enable_runtime_graph: bool = True):
        # ... existing init ...
        self._external_parent_span_id: Optional[str] = None
```

### 2. Update `attach_to_trace`
Capture the `parent_span_id` when attaching to a specific trace context.

```python
    def attach_to_trace(
        self,
        trace_id: str,
        parent_span_id: Optional[str] = None
    ) -> None:
        self._current_trace_id = trace_id
        self._external_parent_span_id = parent_span_id  # <--- Store this
        self._output_attach_to_trace(trace_id, parent_span_id)
```

### 3. Update `on_task_start`
Use `_external_parent_span_id` as a fallback when the local stack is empty.

```python
    def on_task_start(self, task: Executable, context: ExecutionContext) -> None:
        parent_task_id = None
        
        # 1. Check local execution stack (for nested tasks within this process)
        if hasattr(context, 'current_task_id') and context.current_task_id:
            parent_task_id = context.current_task_id
        
        # 2. If no local parent, check external parent (for distributed workers)
        elif self._external_parent_span_id:
            # Only use external parent for the ROOT task of this execution
            # (Subsequent nested tasks in this worker will have a local parent)
            parent_task_id = self._external_parent_span_id

        self.span_start(
            task.task_id,
            parent_name=parent_task_id,
            # ...
        )
```

### 4. Edge Case Handling: Clearing External Parent
Once the first task in the worker has started, we technically "consumed" the external parent for that chain. However, keeping it doesn't hurt because subsequent tasks will see the local `current_task_id` (the first task) on the stack, which takes precedence in the `if/elif` logic.

## Verification
- **Local:** `executing_task` for root task -> Parent None. Nested -> Parent A. (Unchanged behavior).
- **Worker:** `attach_to_trace(parent=P)`. `executing_task(Child)` -> Stack empty -> Fallback P. Edge `P -> Child` created.

This ensures the `_runtime_graph` correctly connects distributed tasks to their originators.
