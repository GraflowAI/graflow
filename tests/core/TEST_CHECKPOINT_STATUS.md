# Checkpoint Tests Status

## Summary

Comprehensive unit and scenario tests have been created for the checkpoint/resume functionality based on `docs/checkpoint_resume_design.md`.

**Current Status**: ✅ **31/31 unit tests passing (100%)** - Architectural fix applied

## Test Files

### 1. `tests/core/test_checkpoint.py` - Unit Tests (31 tests)

**All Tests Passing (31/31)**:
- ✅ CheckpointMetadata tests (4/4)
- ✅ CheckpointManagerBasic tests (7/7)
- ✅ CheckpointCreation tests (7/7)
- ✅ CheckpointRestoration tests (5/5)
- ✅ Error Handling tests (5/5)
- ✅ Integration tests (3/3)

### 2. `tests/scenario/test_checkpoint_scenarios.py` - Scenario Tests (12 tests)

End-to-end workflow tests covering:
- State machine workflows with checkpoints
- Iterative workflows with periodic checkpoints
- Fault tolerance and recovery
- Dynamic task generation
- Complex workflows with cycles and branches
- Checkpoint metadata tracking

**Status**: Not yet run (will have same serialization issue)

## Issue Resolution

### Task Serialization Problem - ✅ FIXED

**Root Cause**: `InMemoryTaskQueue.__init__()` created NEW `Task` objects instead of using existing tasks from the graph. This violated the single source of truth principle and created unnecessary coupling.

**Initial Fix**: Added serialization attributes (`__name__`, `__module__`, `__qualname__`) to both `Task` and `TaskWrapper` classes to make all task objects serializable.

**Architectural Fix (Applied)**: Updated `InMemoryTaskQueue.__init__()` to use tasks from the graph instead of creating new ones:

**File**: `graflow/queue/memory.py`

```python
def __init__(self, execution_context, start_node: Optional[str] = None):
    super().__init__(execution_context)
    self._queue: deque[TaskSpec] = deque()
    if start_node:
        # Get task from graph instead of creating a new Task object
        # This ensures single source of truth and preserves task metadata
        start_task = execution_context.graph.get_node(start_node)

        if start_task is None:
            available_nodes = list(execution_context.graph.nodes.keys())
            raise ValueError(
                f"Start node '{start_node}' not found in graph. "
                f"Available nodes: {available_nodes}"
            )

        task_spec = TaskSpec(
            executable=start_task,
            execution_context=execution_context
        )
        self._queue.append(task_spec)
        self._task_specs[start_node] = task_spec
```

**Benefits**:
- ✅ Single source of truth: Graph owns all tasks
- ✅ No duplication: Queue references graph's tasks
- ✅ Preserves metadata: Full task information available
- ✅ Validates early: Checks start_node exists before execution

**File**: `graflow/core/task.py` (serialization support retained as defense in depth)

```python
# Task.__init__ (lines 302-313) - Serialization attributes added
def __init__(self, task_id: str, register_to_context: bool = True) -> None:
    super().__init__()
    self._task_id = task_id

    # Add serialization attributes required for checkpoint/resume
    self.__name__ = task_id
    self.__module__ = self.__class__.__module__
    self.__qualname__ = f"{self.__class__.__name__}.{task_id}"

    if register_to_context:
        self._register_to_context()

# TaskWrapper.__init__ (lines 610-636) - Serialization attributes added
def __init__(self, task_id: str, func, inject_context: bool = False, ...):
    super().__init__()
    self._task_id = task_id
    self.func = func
    self.inject_context = inject_context

    # Add serialization attributes required for checkpoint/resume
    self.__name__ = getattr(func, '__name__', task_id)
    self.__module__ = getattr(func, '__module__', self.__class__.__module__)
    self.__qualname__ = getattr(func, '__qualname__', task_id)
    self.__doc__ = getattr(func, '__doc__', None)
    ...
```

**Result**: All task objects (Task, TaskWrapper) are now fully serializable.

## Alternative Implementation Options

The following options were considered but not implemented:

### Option 1: Use tasks from graph
```python
# graflow/queue/memory.py
def __init__(self, execution_context, start_node: Optional[str] = None):
    super().__init__(execution_context)
    self._queue: deque[TaskSpec] = deque()
    if start_node:
        # Get task from graph instead of creating new one
        start_task = execution_context.graph.get_node(start_node)
        if start_task:
            task_spec = TaskSpec(
                executable=start_task,
                execution_context=execution_context
            )
            self._queue.append(task_spec)
            self._task_specs[start_node] = task_spec
```

### Option 2: Add serialization support to Task class
```python
# graflow/core/task.py
class Task:
    def __init__(self, task_id: str, ...):
        self.task_id = task_id
        self.__name__ = task_id  # Add for serialization
        self.__module__ = self.__class__.__module__
        self.__qualname__ = f"{self.__class__.__name__}.{task_id}"
```

### Option 3: Use TaskWrapper instead
```python
# graflow/queue/memory.py
from graflow.core.task import TaskWrapper
def __init__(self, execution_context, start_node: Optional[str] = None):
    if start_node:
        # Use TaskWrapper which has serialization support
        start_task = TaskWrapper(start_node, lambda: None)
        # ... rest of the code
```

## API Change

Updated `ExecutionContext.create()` to accept `Optional[str]` for `start_node`:

```python
# graflow/core/context.py
@classmethod
def create(cls, graph: TaskGraph, start_node: Optional[str] = None, ...):
    """Create execution context with optional start_node."""
```

This allows creating contexts without a start node, which is useful for:
- Checkpoint/resume scenarios
- Manual queue management
- Test scenarios

## Test Coverage

The tests comprehensively cover:

✅ **CheckpointMetadata**:
- Creation, serialization, deserialization, roundtrip

✅ **CheckpointManager Basics**:
- ID generation, backend inference, path handling

✅ **Checkpoint Creation**:
- Basic checkpoints, execution state, user metadata
- Pending tasks, current task, auto-path generation

✅ **Checkpoint Restoration**:
- Basic resume, state preservation, pending tasks
- Metadata preservation, flag clearing

✅ **Error Handling**:
- Non-local backends, missing files, corrupted data

✅ **Integration**:
- Simple workflows, multiple checkpoints, channel data

✅ **Scenario Tests**:
- State machines, fault tolerance, dynamic tasks
- Complex workflows, metadata tracking

## Next Steps

1. ✅ **Fixed InMemoryTaskQueue** - Now uses tasks from graph (architectural fix applied)
2. ✅ **All unit tests passing** - 31/31 tests passing after fixing tests to work with architectural changes
3. 🔄 **Run scenario tests** - verify end-to-end checkpoint/resume workflows
4. 🔄 **Integration testing** with real workflows

## Test Updates After Architectural Fix

After applying the architectural fix, tests that created contexts with `start_node="start"` on empty graphs were updated to use `start_node=None`. This ensures tests properly populate the graph before using a start node, or use `None` when the graph is empty.

**Tests Updated**:
- test_create_checkpoint_with_user_metadata
- test_create_checkpoint_auto_path_generation
- test_create_checkpoint_updates_context
- test_resume_metadata_preservation
- test_resume_clears_checkpoint_request
- test_create_checkpoint_nonlocal_backend_error
- test_resume_checkpoint_corrupted_state
- test_resume_checkpoint_corrupted_metadata
- test_multiple_checkpoints_same_session
- test_checkpoint_with_channel_data

## Running Tests

```bash
# Unit tests
PYTHONPATH=. uv run pytest tests/core/test_checkpoint.py -v

# Scenario tests
PYTHONPATH=. uv run pytest tests/scenario/test_checkpoint_scenarios.py -v

# All checkpoint tests
PYTHONPATH=. uv run pytest tests/core/test_checkpoint.py tests/scenario/test_checkpoint_scenarios.py -v

# Specific test
PYTHONPATH=. uv run pytest tests/core/test_checkpoint.py::TestCheckpointCreation::test_create_checkpoint_basic -v
```

## Test Examples

### Unit Test Example
```python
def test_create_checkpoint_basic(self):
    """Test creating a basic checkpoint with minimal state."""
    graph = TaskGraph()

    @task("task_a")
    def task_a_func():
        return "task_a_result"

    graph.add_node(task_a_func, "task_a")
    context = ExecutionContext.create(graph, start_node=None, max_steps=10)
    context.start_node = "task_a"

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint")
        pkl_path, metadata = CheckpointManager.create_checkpoint(
            context, path=checkpoint_path
        )

        assert os.path.exists(pkl_path)
        assert os.path.exists(f"{checkpoint_path}.state.json")
        assert metadata.session_id == context.session_id
```

### Scenario Test Example
```python
def test_state_machine_workflow_with_checkpoints(self):
    """Test state machine workflow that checkpoints at each state transition."""
    graph = TaskGraph()

    @task("process_order", inject_context=True)
    def process_order_func(task_ctx):
        channel = task_ctx.execution_context.channel
        state = channel.get("order_state") or "NEW"

        if state == "NEW":
            # Validate order
            channel.set("order_state", "VALIDATED")
            task_ctx.checkpoint(metadata={"stage": "validation_complete"})
            task_ctx.next_iteration()
        elif state == "VALIDATED":
            # Process payment
            channel.set("order_state", "PAID")
            task_ctx.checkpoint(metadata={"stage": "payment_complete"})
            task_ctx.next_iteration()
        elif state == "PAID":
            # Ship order
            channel.set("order_state", "SHIPPED")
            return "ORDER_COMPLETE"

    # Execute workflow with checkpoints at each state transition
    # Resume from checkpoint after simulated failure
    # Verify state preservation across resume
```
