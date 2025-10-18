# ParallelGroup Error Handling and Custom Handler Design

**Status**: Design Phase
**Created**: 2025-01-19
**Author**: Design Discussion with User

## Summary

This design extends **TaskHandler** to control success criteria for parallel task groups via `on_group_finished()` method.

**Key Points**:
- **Default behavior**: Strict mode - raise exception if ANY task fails (fail-fast)
- **Custom handlers**: Users implement `TaskHandler` with custom `on_group_finished()` logic
- **No built-in policies**: Only default strict handler is built-in; all others are user-implemented
- **Consistent API**: Uses existing `TaskHandler` abstraction - no new handler type needed
- **Breaking change**: Fixes current bug where failures are silently ignored

**Design Philosophy**:
- **Minimal core**: Only `DirectTaskHandler` (strict mode) is built-in to avoid maintenance burden
- **User-driven customization**: Sample handlers in `examples/` for users to copy and adapt
- **Two-point focus**: ① Detect task failures and fail the group, ② Provide hook for custom policies
- **Fail-safe default**: Workflows should fail on task failures by default
- **Library responsibility**: Maintain standard API compatibility only; users own custom policy code
- **Handler as instance**: Handlers passed as instances, not strings - simplifies serialization and avoids registry complexity

## Table of Contents

1. [Background and Motivation](#background-and-motivation)
2. [Current Problems](#current-problems)
3. [Design Overview](#design-overview)
4. [API Design](#api-design)
5. [Distributed Execution Considerations](#distributed-execution-considerations)
6. [Implementation Plan](#implementation-plan)
7. [Usage Examples](#usage-examples)
8. [Backward Compatibility](#backward-compatibility)
9. [Future Extensions](#future-extensions)

---

## Background and Motivation

### Current State

Graflow supports parallel task execution via `ParallelGroup`:

```python
(task_a | task_b | task_c | task_d).with_execution(
    backend=CoordinationBackend.THREADING
)
```

However, the current implementation has **critical error handling issues**:

1. **Task-level handlers** (`TaskHandler`) control HOW tasks execute (direct, docker, logging)
2. **No group-level handlers** to control WHEN a parallel group succeeds or fails
3. **Failures are silently ignored** in coordinators (threading.py, redis.py)

### Motivation

Users need flexible control over parallel group success criteria:

- **Strict mode** (default): All tasks must succeed - raises exception on any failure
- **Custom logic**: Users implement their own handlers for specific requirements
  - "Continue even if some tasks fail" (best-effort)
  - "Success if at least 3 out of 4 tasks succeed"
  - "Success if critical tasks succeed, others optional"
  - Any custom success criteria via `TaskHandler.on_group_finished()` implementation

---

## Current Problems

### Problem 1: Task Failures Are Ignored

**File**: `graflow/coordination/threading.py:75-83`

```python
if success:
    success_count += 1
    execution_context.merge_results(branch_context)
else:
    failure_count += 1
    print(f"  ✗ Task {task_id} failed: {message}")
    # ← No exception raised!

print(f"  Threading group {group_id} completed: {success_count} success, {failure_count} failed")
# ← Always treated as success
```

**Impact**: Workflow continues even when parallel tasks fail, leading to incorrect results.

### Problem 2: No Customization for Success Criteria

Users cannot specify:
- "At least 3 out of 4 tasks must succeed"
- "All critical tasks must succeed, others are optional"
- "Continue on any failure (best-effort)"

### Problem 3: Inconsistent with TaskHandler Pattern

Graflow already has a handler pattern for tasks:

```python
@task(handler="docker")
def my_task():
    pass

engine.register_handler("docker", DockerTaskHandler())
```

But there's no equivalent for parallel groups.

---

## Design Overview

### Unified TaskHandler Architecture

```
┌──────────────────────────────────────────────────────────────┐
│ TaskHandler (unified handler for both levels)               │
│                                                              │
│  execute_task(task, context)                                │
│    - Controls HOW each task executes                        │
│    - Examples: direct execution, docker, logging            │
│                                                              │
│  on_group_finished(group_id, tasks, results, context) │
│    - Controls WHEN parallel groups succeed/fail             │
│    - Default: strict mode (all tasks must succeed)          │
│    - Override: custom success criteria                      │
│                                                              │
│  Specified: @task(handler="custom")                         │
│            .with_execution(handler="custom")                │
│            .with_execution(handler=CustomHandler())         │
└──────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Unified abstraction**: Single TaskHandler for both task execution and group policy
2. **Composition over inheritance**: Use composition pattern to separate concerns
3. **Fail-safe default**: Strict mode (all tasks must succeed)
4. **Backward compatible**: Existing handlers work without modification
5. **Distributed-aware**: Works with both local and Redis execution

### Addressing Codex Concerns: Separation of Concerns

**Concern**: Combining execution strategy (Docker, etc.) with success policy in one handler reduces reusability.

**Solution**: Use **composition pattern** to delegate responsibilities:

```python
# Execution-only handlers (unchanged from current)
class DirectTaskHandler(TaskHandler):
    def get_name(self) -> str:
        return "direct"

    def execute_task(self, task, context):
        result = task.run()
        context.set_result(task.task_id, result)
    # Inherits default strict on_group_finished()

# Policy-only handlers (focus on group success criteria)
class AtLeastNSuccessHandler(TaskHandler):
    def __init__(self, min_success: int):
        self.min_success = min_success

    def get_name(self) -> str:
        return f"at_least_{self.min_success}"

    def execute_task(self, task, context):
        # Delegate to direct execution
        result = task.run()
        context.set_result(task.task_id, result)

    def on_group_finished(self, group_id, tasks, results, context):
        # Custom policy logic
        ...

# Composition pattern: Combine execution strategy + policy
class PolicyDelegatingHandler(TaskHandler):
    """Delegates execution to one handler, group policy to another."""

    def __init__(self, execution_handler: TaskHandler, policy_handler: TaskHandler):
        self.execution_handler = execution_handler
        self.policy_handler = policy_handler

    def get_name(self) -> str:
        return f"{self.execution_handler.get_name()}_{self.policy_handler.get_name()}"

    def execute_task(self, task, context):
        # Delegate to execution handler
        return self.execution_handler.execute_task(task, context)

    def on_group_finished(self, group_id, tasks, results, context):
        # Delegate to policy handler
        return self.policy_handler.on_group_finished(group_id, tasks, results, context)

# Usage: Combine Docker execution with "at least 3" policy
docker_at_least_3 = PolicyDelegatingHandler(
    execution_handler=DockerTaskHandler(),
    policy_handler=AtLeastNSuccessHandler(min_success=3)
)
engine.register_handler("docker_at_least_3", docker_at_least_3)

(task_a | task_b | task_c | task_d).with_execution(handler="docker_at_least_3")
```

**Benefits**:
- **Separation**: Execution strategy and success policy are separate objects
- **Reusability**: Any execution handler × Any policy handler
- **No inheritance explosion**: Composition, not inheritance
- **Backward compatible**: Existing handlers continue to work

---

## API Design

### 1. TaskHandler Base Class (Updated)

**File**: `graflow/core/handler.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class TaskResult:
    """Result of a task execution."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    duration: float = 0.0
    timestamp: float = 0.0


class TaskHandler(ABC):
    """Base class for task execution handlers.

    TaskHandler controls both:
    1. How individual tasks execute (execute_task)
    2. When parallel groups succeed/fail (on_group_finished)
    """

    def get_name(self) -> str:
        """Get handler name for registration.

        Returns:
            Handler name (used for registration and lookup)

        Note:
            This method has a default implementation for backward compatibility.
            Override to provide a custom name.

        Examples:
            >>> class DirectTaskHandler(TaskHandler):
            ...     def get_name(self):
            ...         return "direct"
            >>>
            >>> class AtLeastNSuccessHandler(TaskHandler):
            ...     def __init__(self, min_success: int):
            ...         self.min_success = min_success
            ...     def get_name(self):
            ...         return f"at_least_{self.min_success}"
            >>>
            >>> # Default implementation (no override needed)
            >>> class MyHandler(TaskHandler):
            ...     # get_name() inherited -> returns "MyHandler"
            ...     def execute_task(self, task, context): ...
        """
        # Default implementation: Use class name
        return self.__class__.__name__

    @abstractmethod
    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """Execute single task and store result in context.

        Args:
            task: Executable task to execute
            context: Execution context

        Note:
            Implementation must call context.set_result(task_id, result) or
            context.set_result(task_id, exception) within the execution environment.
        """
        pass

    def on_group_finished(
        self,
        group_id: str,
        tasks: List[Executable],
        results: Dict[str, TaskResult],
        context: ExecutionContext
    ) -> None:
        """Handle parallel group execution results.

        Default: Strict mode - fail if any task fails.
        Override for custom success criteria.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks in the group
            results: Dict mapping task_id to TaskResult
            context: Execution context

        Raises:
            ParallelGroupError: If group execution should fail

        Note:
            Called after ALL tasks complete (success or failure).
            Handler decides whether to raise exception based on results.
        """
        # Default implementation: Strict mode
        from graflow.exceptions import ParallelGroupError

        failed_tasks = [
            (task_id, result.error_message or "Unknown error")
            for task_id, result in results.items()
            if not result.success
        ]

        if failed_tasks:
            raise ParallelGroupError(
                f"Parallel group {group_id} failed: {len(failed_tasks)} task(s) failed",
                group_id=group_id,
                failed_tasks=failed_tasks,
                successful_tasks=[tid for tid, r in results.items() if r.success]
            )
```

### 2. Built-in Handler: DirectTaskHandler (Updated)

**File**: `graflow/core/handlers/direct.py`

```python
class DirectTaskHandler(TaskHandler):
    """Direct task execution handler (default).

    Executes tasks directly in the current process.
    Uses default strict mode for group execution (inherited from TaskHandler).
    """

    def get_name(self) -> str:
        """Return handler name."""
        return "direct"

    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """Execute task directly."""
        try:
            result = task.run()
            context.set_result(task.task_id, result)
        except Exception as e:
            context.set_result(task.task_id, e)
            raise

    # on_group_finished is inherited from TaskHandler (strict mode)
```

Note: `DirectTaskHandler` inherits the default strict mode from `TaskHandler.on_group_finished()`.

### 3. Custom Handler Examples

Users can implement custom handlers for any success criteria.

**Example 1: Best-effort handler (never fail)**
```python
class BestEffortHandler(TaskHandler):
    """Continue even if tasks fail."""

    def get_name(self) -> str:
        return "best_effort"

    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """Execute task (delegates to direct execution)."""
        try:
            result = task.run()
            context.set_result(task.task_id, result)
        except Exception as e:
            context.set_result(task.task_id, e)
            raise

    def on_group_finished(self, group_id, tasks, results, context):
        """Never fail - always succeed."""
        failed = [tid for tid, r in results.items() if not r.success]
        if failed:
            print(f"⚠️  Group {group_id} completed with {len(failed)} failures (best-effort)")
        # Don't raise exception - always succeed
```

**Example 2: At least N success handler**
```python
class AtLeastNSuccessHandler(TaskHandler):
    """Need at least N tasks to succeed."""

    def __init__(self, min_success: int):
        self.min_success = min_success

    def get_name(self) -> str:
        return f"at_least_{self.min_success}"

    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """Execute task (delegates to direct execution)."""
        try:
            result = task.run()
            context.set_result(task.task_id, result)
        except Exception as e:
            context.set_result(task.task_id, e)
            raise

    def on_group_finished(self, group_id, tasks, results, context):
        """Require at least N tasks to succeed."""
        successful = [tid for tid, r in results.items() if r.success]

        if len(successful) < self.min_success:
            failed = [(tid, r.error_message) for tid, r in results.items() if not r.success]
            raise ParallelGroupError(
                f"Only {len(successful)}/{self.min_success} tasks succeeded",
                group_id=group_id,
                failed_tasks=failed,
                successful_tasks=successful
            )
```

**Example 3: Threshold handler**
```python
class ThresholdHandler(TaskHandler):
    """Need X% success rate."""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def get_name(self) -> str:
        return f"threshold_{int(self.threshold * 100)}"

    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """Execute task (delegates to direct execution)."""
        try:
            result = task.run()
            context.set_result(task.task_id, result)
        except Exception as e:
            context.set_result(task.task_id, e)
            raise

    def on_group_finished(self, group_id, tasks, results, context):
        """Require threshold percentage to succeed."""
        if not results:
            raise ParallelGroupError("No results available")

        success_rate = len([r for r in results.values() if r.success]) / len(results)

        if success_rate < self.threshold:
            failed = [(tid, r.error_message) for tid, r in results.items() if not r.success]
            raise ParallelGroupError(
                f"Success rate {success_rate:.1%} < {self.threshold:.1%}",
                group_id=group_id,
                failed_tasks=failed,
                successful_tasks=[tid for tid, r in results.items() if r.success]
            )
```

**Example 4: Critical tasks handler**
```python
class CriticalTasksHandler(TaskHandler):
    """Succeed if critical tasks succeed, ignore others."""

    def __init__(self, critical_task_ids: List[str]):
        self.critical_task_ids = set(critical_task_ids)

    def get_name(self) -> str:
        return "critical_tasks"

    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """Execute task (delegates to direct execution)."""
        try:
            result = task.run()
            context.set_result(task.task_id, result)
        except Exception as e:
            context.set_result(task.task_id, e)
            raise

    def on_group_finished(self, group_id, tasks, results, context):
        """Succeed if critical tasks succeed."""
        failed_critical = [
            tid for tid in self.critical_task_ids
            if tid in results and not results[tid].success
        ]

        if failed_critical:
            raise ParallelGroupError(
                f"Critical tasks failed: {failed_critical}",
                group_id=group_id,
                failed_tasks=[(tid, results[tid].error_message) for tid in failed_critical],
                successful_tasks=[tid for tid, r in results.items() if r.success]
            )
```

Note: These examples show how to implement custom handlers. Only `DirectTaskHandler` is built-in.

### 4. ParallelGroupError Exception

**File**: `graflow/exceptions.py`

```python
class ParallelGroupError(GraflowRuntimeError):
    """Exception raised when parallel group execution fails.

    Attributes:
        group_id: Parallel group identifier
        failed_tasks: List of (task_id, error_message) tuples
        successful_tasks: List of successful task IDs
    """

    def __init__(
        self,
        message: str,
        group_id: str,
        failed_tasks: List[tuple[str, str]],
        successful_tasks: List[str]
    ):
        super().__init__(message)
        self.group_id = group_id
        self.failed_tasks = failed_tasks
        self.successful_tasks = successful_tasks
```

### 5. WorkflowEngine Handler Registry (No Changes)

**File**: `graflow/core/engine.py`

No changes needed to WorkflowEngine! The existing handler registry works for both task and group handlers.

```python
class WorkflowEngine:
    def __init__(self):
        self._handlers: Dict[str, TaskHandler] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        # Register default direct handler
        from graflow.core.handlers.direct import DirectTaskHandler
        self._handlers['direct'] = DirectTaskHandler()

    def register_handler(self, handler_type: str, handler: TaskHandler):
        """Register a custom handler (existing method - no changes)."""
        self._handlers[handler_type] = handler

    def get_handler(self, handler_type: str) -> TaskHandler:
        """Get handler by name (existing method - no changes)."""
        if handler_type not in self._handlers:
            raise ValueError(
                f"Handler '{handler_type}' not registered. "
                f"Available: {list(self._handlers.keys())}"
            )
        return self._handlers[handler_type]
```

The beauty of this design: **No API changes to WorkflowEngine!** The existing handler registry automatically supports both task and group execution.

### 6. ParallelGroup.with_execution() API

**File**: `graflow/core/task.py`

```python
class ParallelGroup(Executable):
    def with_execution(
        self,
        backend: Optional[CoordinationBackend] = None,
        backend_config: Optional[dict] = None,
        handler: Optional[TaskHandler] = None
    ) -> ParallelGroup:
        """Configure execution backend and handler.

        Args:
            backend: Coordinator backend (DIRECT, THREADING, REDIS)
            backend_config: Backend-specific configuration
            handler: TaskHandler instance for group execution
                    - TaskHandler instance: Passed directly to coordinator
                    - None (default): Uses DirectTaskHandler (strict mode - all tasks must succeed)

        Returns:
            Self (for method chaining)

        Examples:
            # Default: All tasks must succeed (strict mode)
            (task_a | task_b | task_c).with_execution()

            # Best-effort: Continue even if tasks fail (custom handler)
            (task_a | task_b | task_c).with_execution(
                handler=BestEffortHandler()
            )

            # At least 3 out of 4 tasks must succeed
            (task_a | task_b | task_c | task_d).with_execution(
                handler=AtLeastNSuccessHandler(min_success=3)
            )

            # Critical tasks must succeed
            (task_a | task_b).with_execution(
                handler=CriticalTasksHandler(critical_task_ids=["task_a"])
            )
        """
        # Store configuration
        if handler is not None:
            self._execution_config["handler"] = handler
        # ...
        return self
```

### 7. TaskCoordinator Interface Update

**File**: `graflow/coordination/coordinator.py`

```python
class TaskCoordinator(ABC):
    @abstractmethod
    def execute_group(
        self,
        group_id: str,
        tasks: List[Executable],
        execution_context: ExecutionContext,
        handler: Optional[TaskHandler] = None  # ← New parameter
    ) -> None:
        """Execute parallel group with handler.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            execution_context: Execution context
            handler: TaskHandler instance for group execution (None = use default DirectTaskHandler)
        """
        pass
```

---

## Data Flow: TaskResult Generation and Aggregation

### Addressing Codex Concern: TaskResult Lifecycle

**Concern**: Who creates `Dict[str, TaskResult]`, when, and how is it collected without data loss in distributed execution?

**Solution**: Clear data flow at each execution layer.

### Local Execution (Direct/Threading)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Task Execution (in thread/process)                       │
│    try:                                                      │
│        result = task.run()                                   │
│        context.set_result(task_id, result)                   │
│    except Exception as e:                                    │
│        context.set_result(task_id, e)                        │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Coordinator Collects Results (ThreadingCoordinator)      │
│                                                              │
│    results: Dict[str, TaskResult] = {}                      │
│                                                              │
│    for future in futures:                                   │
│        task_id, success, message = future.result()          │
│                                                              │
│        # Create TaskResult from execution outcome           │
│        results[task_id] = TaskResult(                        │
│            task_id=task_id,                                  │
│            success=success,                                  │
│            error_message=message if not success else None,   │
│            timestamp=time.time()                             │
│        )                                                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Handler Processes Results                                │
│                                                              │
│    handler.on_group_finished(                           │
│        group_id, tasks, results, context                     │
│    )                                                         │
└─────────────────────────────────────────────────────────────┘
```

**Implementation** (graflow/coordination/threading.py):
```python
def execute_group(self, group_id, tasks, execution_context, handler=None):
    # ... task execution ...

    # Collect results
    results: Dict[str, TaskResult] = {}
    for future in completed_futures:
        task_id, success, message = future.result()

        # Create TaskResult
        results[task_id] = TaskResult(
            task_id=task_id,
            success=success,
            error_message=message if not success else None,
            timestamp=time.time()
        )

    # Apply handler
    handler.on_group_finished(group_id, tasks, results, execution_context)
```

### Distributed Execution (Redis)

```
┌─────────────────────────────────────────────────────────────┐
│ Worker Process                                               │
│                                                              │
│  1. Execute task                                             │
│     try:                                                     │
│         result = task.run()                                  │
│         success = True                                       │
│         error_message = None                                 │
│     except Exception as e:                                   │
│         success = False                                      │
│         error_message = str(e)                               │
│                                                              │
│  2. Store in Redis (record_task_completion)                 │
│     completion_data = {                                      │
│         "task_id": task_id,                                  │
│         "success": success,                                  │
│         "error_message": error_message,                      │
│         "timestamp": time.time()                             │
│     }                                                        │
│     redis.zadd(f"completions:{group_id}", {json.dumps(...)})│
└─────────────────────────────────────────────────────────────┘
                           ↓ Redis
┌─────────────────────────────────────────────────────────────┐
│ Coordinator Process (RedisCoordinator)                       │
│                                                              │
│  3. Wait for barrier                                         │
│     wait_barrier(group_id)                                   │
│                                                              │
│  4. Collect results from Redis                              │
│     completion_records = redis.zrange(f"completions:{group_id}")│
│                                                              │
│     results: Dict[str, TaskResult] = {}                     │
│     for record in completion_records:                        │
│         data = json.loads(record)                            │
│         results[data["task_id"]] = TaskResult(               │
│             task_id=data["task_id"],                         │
│             success=data["success"],                         │
│             error_message=data.get("error_message"),         │
│             timestamp=data.get("timestamp")                  │
│         )                                                    │
│                                                              │
│  5. Apply handler                                            │
│     handler.on_group_finished(group_id, tasks, results, context)│
└─────────────────────────────────────────────────────────────┘
```

**Implementation** (graflow/coordination/redis.py):
```python
def execute_group(self, group_id, tasks, execution_context, handler=None):
    # Dispatch tasks
    self.create_barrier(group_id, len(tasks))
    for task in tasks:
        self.dispatch_task(task, group_id)

    # Wait for all tasks
    if not self.wait_barrier(group_id):
        raise TimeoutError(f"Barrier timeout: {group_id}")

    # Collect results from Redis
    completion_results = self._get_completion_results(group_id)

    # Convert to TaskResult
    task_results: Dict[str, TaskResult] = {}
    for result_data in completion_results:
        task_results[result_data["task_id"]] = TaskResult(
            task_id=result_data["task_id"],
            success=result_data["success"],
            error_message=result_data.get("error_message"),
            timestamp=result_data.get("timestamp", 0.0)
        )

    # Apply handler
    handler.on_group_finished(group_id, tasks, task_results, execution_context)

def _get_completion_results(self, group_id: str) -> List[Dict[str, Any]]:
    """Retrieve all completion records from Redis."""
    completion_key = f"{self.task_queue.key_prefix}:completions:{group_id}"
    records = self.redis.zrange(completion_key, 0, -1)

    results = []
    for record in records:
        try:
            data = json.loads(record)
            results.append(data)
        except json.JSONDecodeError:
            continue

    return results
```

### Data Loss Prevention

**Guarantee**: All task completions are captured before handler execution.

1. **Local execution**: Futures API ensures no results are lost
2. **Distributed execution**: Redis sorted set ensures atomic result storage
3. **Barrier synchronization**: Ensures all tasks complete before result collection

**Edge cases handled**:
- Worker crash: Task remains in RUNNING state, barrier times out → Group fails
- Network partition: Redis atomic operations prevent partial updates
- Duplicate completion: Sorted set + task_id as key prevents duplicates

---

## Handler Passing and Serialization Strategy

### Design Decision: Handler Instances Only

**Simplified Approach**: Handlers are passed as **instances**, not strings. This eliminates the need for:
- Handler registry in ExecutionContext
- Engine reference in ExecutionContext
- Auto-registration logic
- Serialization complexity

**Benefits**:
- ✅ Simple and explicit
- ✅ No serialization issues (handler instance is pickled with ExecutionContext)
- ✅ No registry management overhead
- ✅ Clear ownership: user creates and passes handler instance

### Handler Lifecycle Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Configuration Phase (Workflow Definition)                    │
│                                                                  │
│    handler = AtLeastNSuccessHandler(min_success=3)             │
│    (task_a | task_b | task_c | task_d).with_execution(         │
│        handler=handler                                          │
│    )                                                             │
│          ↓                                                       │
│    ParallelGroup._execution_config["handler"] = handler         │
│                                                                  │
│    Note: Handler instance is STORED in config                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Execution Phase (ParallelGroup.run())                        │
│                                                                  │
│    def run(self):                                                │
│        context = self.get_execution_context()                   │
│        handler = self._execution_config.get("handler")          │
│                                                                  │
│        # Create executor with backend config                    │
│        executor = self._create_configured_executor()            │
│                                                                  │
│        # Pass handler instance to execute_parallel_group        │
│        executor.execute_parallel_group(                         │
│            group_id=self.task_id,                               │
│            tasks=self.tasks,                                    │
│            exec_context=context,                                │
│            handler=handler  # ← Handler instance (or None)      │
│        )                                                         │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Handler Resolution (GroupExecutor)                           │
│                                                                  │
│    def execute_parallel_group(                                  │
│        self, group_id, tasks, exec_context,                     │
│        handler=None  # ← Optional TaskHandler instance          │
│    ):                                                            │
│        # Resolve handler                                        │
│        if handler is None:                                      │
│            # Create default handler (strict mode)               │
│            from graflow.core.handlers.direct import DirectTaskHandler│
│            handler = DirectTaskHandler()                        │
│                                                                  │
│        # Pass handler to coordinator                            │
│        if self.backend == CoordinationBackend.DIRECT:           │
│            self.direct_execute(group_id, tasks, exec_context, handler)│
│        else:                                                     │
│            coordinator = self._create_coordinator(...)          │
│            coordinator.execute_group(                           │
│                group_id, tasks, exec_context, handler           │
│            )                                                     │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Coordinator Execution                                         │
│                                                                  │
│    coordinator.execute_group(                                   │
│        group_id, tasks, execution_context, handler              │
│    )                                                             │
│                                                                  │
│    # Coordinator uses handler instance directly                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. Handlers Are Instances, Not Strings

**Design**: Handlers are passed as `TaskHandler` instances, not string names.

**Why**:
- **Simplicity**: No registry management needed
- **Explicitness**: User creates and controls handler instance
- **Serializability**: Handler instance is pickled with ParallelGroup config
- **No engine dependency**: ExecutionContext doesn't need engine reference

**File**: `graflow/coordination/executor.py`

```python
class GroupExecutor:
    """Unified executor for parallel task groups supporting multiple backends."""

    def __init__(
        self,
        backend: CoordinationBackend = CoordinationBackend.THREADING,
        backend_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize GroupExecutor with backend configuration."""
        self.backend = backend
        self.backend_config = backend_config or {}

    def execute_parallel_group(
        self,
        group_id: str,
        tasks: List[Executable],
        exec_context: ExecutionContext,
        handler: Optional[TaskHandler] = None  # ← Handler instance or None
    ):
        """Execute parallel group with handler.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            exec_context: Execution context
            handler: TaskHandler instance (None = use default DirectTaskHandler)
        """
        # Use handler instance directly or create default
        if handler is None:
            from graflow.core.handlers.direct import DirectTaskHandler
            handler = DirectTaskHandler()

        # Execute with appropriate backend
        if self.backend == CoordinationBackend.DIRECT:
            self.direct_execute(group_id, tasks, exec_context, handler)
        else:
            coordinator = self._create_coordinator(
                self.backend, self.backend_config, exec_context
            )
            coordinator.execute_group(
                group_id, tasks, exec_context, handler
            )
```

#### 2. Handler Serialization for Distributed Execution

**Challenge**: Handler instances must be serializable for distributed execution.

**Solution**: Handler instances are **pickled** along with `ParallelGroup._execution_config`.

**Requirements for Custom Handlers**:
1. Handler class must be importable on worker side
2. Handler instance must be picklable (no unpicklable state like locks, connections)
3. Simple handlers (stateless or with simple config) work automatically

**Example - Picklable Handler**:
```python
class AtLeastNSuccessHandler(TaskHandler):
    def __init__(self, min_success: int):
        self.min_success = min_success  # Simple int - picklable

    def execute_task(self, task, context):
        # ...

    def on_group_finished(self, group_id, tasks, results, context):
        # ...
```

**Example - Non-Picklable Handler (Avoid)**:
```python
class DatabaseHandler(TaskHandler):
    def __init__(self, db_connection):
        self.db_connection = db_connection  # ❌ Not picklable!
```

**Best Practice for Distributed Execution**:
```python
class DatabaseHandler(TaskHandler):
    def __init__(self, db_url: str):
        self.db_url = db_url  # ✅ Picklable string
        self._connection = None  # Lazy initialization

    def _get_connection(self):
        if self._connection is None:
            self._connection = connect(self.db_url)
        return self._connection
```

---

## Distributed Execution Considerations

### Challenge: Worker-Coordinator Communication

In distributed execution, tasks run in separate worker processes:

```
Coordinator (Main)          Worker 1              Worker 2
     |                         |                     |
     |--dispatch(task_a)------>|                     |
     |--dispatch(task_b)---------------------->|     |
     |                         |                     |
     |  wait_barrier()         |                     |
     |<--------result----------| (success/failure)   |
     |<-----------------------------result-----------| (success/failure)
     |                         |                     |
     | apply group_handler     |                     |
     | (check success criteria)|                     |
```

### Solution: Redis-based Result Aggregation

#### Step 1: Worker Records Completion with Error Info

**File**: `graflow/worker/worker.py`

```python
def _task_completed(self, task_spec: TaskSpec, future: Future):
    result = future.result()
    success = result.get("success", False)
    error_message = result.get("error")

    # Notify with error message
    if task_spec.group_id:
        self.queue.notify_task_completion(
            task_id,
            success,
            task_spec.group_id,
            error_message=error_message  # ← New
        )
```

#### Step 2: RedisTaskQueue Stores Completion Info

**File**: `graflow/queue/redis.py`

```python
def notify_task_completion(
    self,
    task_id: str,
    success: bool,
    group_id: Optional[str] = None,
    error_message: Optional[str] = None  # ← New
):
    if group_id:
        from graflow.coordination.redis import record_task_completion
        record_task_completion(
            self.redis_client,
            self.key_prefix,
            task_id,
            group_id,
            success,
            error_message  # ← New
        )
```

#### Step 3: RedisCoordinator Collects and Applies Handler

**File**: `graflow/coordination/redis.py`

```python
class RedisCoordinator(TaskCoordinator):
    def execute_group(
        self,
        group_id: str,
        tasks: List[Executable],
        execution_context: ExecutionContext,
        handler: Optional[TaskHandler] = None
    ):
        # Default handler
        if handler is None:
            handler = execution_context.engine.get_handler("direct")

        # Dispatch all tasks
        self.create_barrier(group_id, len(tasks))
        for task in tasks:
            self.dispatch_task(task, group_id)

        # Wait for completion
        if not self.wait_barrier(group_id):
            raise TimeoutError(f"Barrier timeout: {group_id}")

        # Collect results from Redis
        completion_results = self._get_completion_results(group_id)

        # Convert to TaskResult format
        task_results = {}
        for result_data in completion_results:
            task_id = result_data["task_id"]
            task_results[task_id] = TaskResult(
                task_id=task_id,
                success=result_data["success"],
                error_message=result_data.get("error_message"),
                timestamp=result_data.get("timestamp", 0.0)
            )

        # Apply handler's group execution logic
        handler.on_group_finished(group_id, tasks, task_results, execution_context)
```

#### Step 4: Helper Function for Redis Storage

**File**: `graflow/coordination/redis.py`

```python
def record_task_completion(
    redis_client: Redis,
    key_prefix: str,
    task_id: str,
    group_id: str,
    success: bool,
    error_message: Optional[str] = None
):
    """Record task completion in Redis for barrier tracking."""
    completion_key = f"{key_prefix}:completions:{group_id}"

    completion_data = {
        "task_id": task_id,
        "success": success,
        "timestamp": time.time(),
        "error_message": error_message
    }

    # Add to sorted set with timestamp as score
    redis_client.zadd(
        completion_key,
        {json.dumps(completion_data): time.time()}
    )
```

---

## Exception Handling Integration

### Addressing Codex Concern: ParallelGroupError and GraflowRuntimeError

**Concern**: In `WorkflowEngine.execute()` (engine.py:108-116), all exceptions are wrapped with `GraflowRuntimeError`. Should `ParallelGroupError` be raised transparently or wrapped? What's the impact on successor task scheduling?

**Solution**: `ParallelGroupError` is **transparently propagated** as a subclass of `GraflowRuntimeError`, terminating workflow execution and preventing successor scheduling.

### Exception Hierarchy

```python
# graflow/exceptions.py

class GraflowRuntimeError(Exception):
    """Base exception for all Graflow runtime errors."""
    pass


class ParallelGroupError(GraflowRuntimeError):
    """Exception raised when parallel group execution fails.

    This is a subclass of GraflowRuntimeError, so it's compatible
    with existing exception handling while providing specific
    information about parallel group failures.
    """

    def __init__(
        self,
        message: str,
        group_id: str,
        failed_tasks: List[tuple[str, str]],
        successful_tasks: List[str]
    ):
        super().__init__(message)
        self.group_id = group_id
        self.failed_tasks = failed_tasks
        self.successful_tasks = successful_tasks
```

**Design Decision**: `ParallelGroupError` inherits from `GraflowRuntimeError`, so:
- ✅ Caught by existing `except GraflowRuntimeError` handlers
- ✅ Provides specific attributes for debugging (group_id, failed_tasks, successful_tasks)
- ✅ No wrapping needed - already a GraflowRuntimeError

### Exception Flow Through Execution Stack

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Handler Evaluates Group Results                              │
│                                                                  │
│    handler.on_group_finished(group_id, tasks, results, ctx)│
│                                                                  │
│    # Default strict mode:                                        │
│    if any task failed:                                           │
│        raise ParallelGroupError(...)                      │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Coordinator Propagates Exception                             │
│                                                                  │
│    # graflow/coordination/threading.py                          │
│    def execute_group(self, group_id, tasks, context, handler):  │
│        # ... collect results ...                                 │
│                                                                  │
│        # Handler may raise ParallelGroupError            │
│        handler.on_group_finished(...)                       │
│        # ← Exception propagates up (no try/catch)                │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. ParallelGroup.run() Propagates Exception                     │
│                                                                  │
│    # graflow/core/task.py                                       │
│    class ParallelGroup(Executable):                             │
│        def run(self, context: ExecutionContext):                │
│            executor = GroupExecutor(context)                    │
│            executor.execute_parallel_group(...)                 │
│            # ← ParallelGroupError propagates              │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. WorkflowEngine Catches and Converts                          │
│                                                                  │
│    # graflow/core/engine.py (lines 108-116)                     │
│    try:                                                          │
│        with context.executing_task(task):                       │
│            self._execute_task(task, context)                    │
│    except Exception as e:                                       │
│        # ParallelGroupError is already GraflowRuntimeError│
│        # as_runtime_error() checks isinstance and returns as-is │
│        raise exceptions.as_runtime_error(e) from e              │
│                                                                  │
│    # Execution terminates - no successor scheduling              │
└─────────────────────────────────────────────────────────────────┘
```

### Key Implementation: as_runtime_error()

**File**: `graflow/exceptions.py`

```python
def as_runtime_error(e: Exception) -> GraflowRuntimeError:
    """Convert exception to GraflowRuntimeError.

    Args:
        e: Exception to convert

    Returns:
        GraflowRuntimeError instance (original or wrapped)

    Note:
        If exception is already a GraflowRuntimeError (including subclasses
        like ParallelGroupError), returns it unchanged.
    """
    if isinstance(e, GraflowRuntimeError):
        # Already a GraflowRuntimeError (or subclass) - return as-is
        return e

    # Wrap other exceptions
    return GraflowRuntimeError(f"Task execution failed: {str(e)}") from e
```

**Behavior**:
- `ParallelGroupError` → Returned as-is (no wrapping)
- Other exceptions → Wrapped in `GraflowRuntimeError`

### Impact on Successor Task Scheduling

**Current behavior** (engine.py:108-138):

```python
def execute(self, context: ExecutionContext, start_task_id: Optional[str] = None):
    while task_id is not None and context.steps < context.max_steps:
        context.reset_goto_flag()

        # Execute task
        try:
            with context.executing_task(task):
                self._execute_task(task, context)  # ← May raise ParallelGroupError
        except Exception as e:
            # Exception stored in context, then re-raised
            raise exceptions.as_runtime_error(e) from e

        # ← This code is NOT reached if exception is raised

        # Increment step
        context.increment_step()

        # Schedule successors (NOT executed if exception raised)
        if not context.goto_called:
            successors = list(graph.successors(task_id))
            for succ in successors:
                context.add_to_queue(succ_task)

        task_id = context.get_next_task()
```

**Behavior when ParallelGroupError is raised**:

1. ✅ Exception propagates through `_execute_task()` → `execute_task()` → `run()`
2. ✅ Caught by `except Exception as e` in engine.py:114
3. ✅ Converted by `as_runtime_error()` (returned as-is since it's already GraflowRuntimeError)
4. ✅ Re-raised, terminating execution loop
5. ❌ **Successors are NOT scheduled** - lines 124-136 are skipped
6. ❌ **Workflow execution terminates** immediately

**Impact Summary**:

| Event | Behavior |
|-------|----------|
| ParallelGroup fails | `ParallelGroupError` raised |
| Exception wrapping | None - already `GraflowRuntimeError` |
| Successor scheduling | ❌ Skipped - execution terminates |
| Exception attributes | ✅ Preserved (group_id, failed_tasks, successful_tasks) |
| Workflow result | ❌ Failed - exception propagates to caller |

### Exception Attributes for Debugging

`ParallelGroupError` provides rich debugging information:

```python
try:
    engine.execute(exec_context)
except ParallelGroupError as e:
    print(f"Group {e.group_id} failed")
    print(f"Failed tasks: {e.failed_tasks}")
    print(f"Successful tasks: {e.successful_tasks}")

    # Example output:
    # Group parallel_group_abc123 failed
    # Failed tasks: [('task_b', 'Connection timeout'), ('task_d', 'Invalid input')]
    # Successful tasks: ['task_a', 'task_c']
```

### Consistency with Existing Exception Handling

**Single task failure** (current behavior):
```python
@task
def failing_task():
    raise ValueError("Task failed")

try:
    engine.execute(exec_context)
except GraflowRuntimeError as e:
    # ValueError wrapped in GraflowRuntimeError
    # No successor scheduling
```

**Parallel group failure** (new behavior):
```python
(task_a | task_b | task_c).with_execution()

try:
    engine.execute(exec_context)
except ParallelGroupError as e:
    # ParallelGroupError (subclass of GraflowRuntimeError)
    # No successor scheduling
    # Additional attributes: group_id, failed_tasks, successful_tasks
```

**Consistency**: Both behave identically from the engine's perspective:
1. Exception raised during task execution
2. Caught by engine, converted to GraflowRuntimeError (or subclass)
3. Re-raised, terminating execution
4. Successors not scheduled

### Design Rationale

**Why subclass instead of wrapping?**

Option A (Wrapping - NOT chosen):
```python
# ParallelGroupError as separate exception
try:
    handler.on_group_finished(...)
except ParallelGroupError as e:
    # Engine wraps it
    raise GraflowRuntimeError(str(e)) from e
    # ❌ Loses group_id, failed_tasks, successful_tasks
```

Option B (Subclass - CHOSEN):
```python
# ParallelGroupError extends GraflowRuntimeError
class ParallelGroupError(GraflowRuntimeError):
    ...

try:
    handler.on_group_finished(...)
except Exception as e:
    # Engine's as_runtime_error() returns it unchanged
    raise exceptions.as_runtime_error(e) from e
    # ✅ Preserves all attributes
    # ✅ Compatible with existing handlers
```

**Benefits of subclass approach**:
- ✅ Preserves debugging information (group_id, failed_tasks)
- ✅ Compatible with existing `except GraflowRuntimeError` handlers
- ✅ Allows specific handling: `except ParallelGroupError`
- ✅ No special-casing in engine.py
- ✅ Consistent with Python exception hierarchy best practices

---

## Implementation Plan

### Phase 1: Core Infrastructure

**Files to modify**:

1. ✅ `graflow/core/handler.py`
   - Add `TaskResult` dataclass
   - Add `get_name()` abstract method to `TaskHandler`
   - Add `on_group_finished()` method to `TaskHandler` with default strict implementation

2. ✅ `graflow/exceptions.py`
   - Add `ParallelGroupError` exception

3. ✅ `graflow/core/handlers/direct.py`
   - Add `get_name()` method to `DirectTaskHandler` returning "direct"
   - Inherit default `on_group_finished()` from `TaskHandler` (strict mode)

4. ✅ `graflow/core/engine.py`
   - **No changes needed** - existing handler registry already supports TaskHandler

### Phase 2: API Surface

**Files to modify**:

5. ✅ `graflow/core/task.py`
   - Update `ParallelGroup._execution_config` to include `handler` field
   - Update `ParallelGroup.with_execution()` signature to accept `handler: Optional[TaskHandler]`
   - Update `ParallelGroup.run()` to:
     - Extract `handler` from `_execution_config`
     - Pass `handler` as 4th parameter to `executor.execute_parallel_group(group_id, tasks, context, handler)`
   - Note: No changes to `_create_configured_executor()` - maintains current `GroupExecutor(backend, backend_config)` API
   - Note: Handler instance is stored in config and pickled with ParallelGroup

### Phase 3: Coordinator Updates

**Files to modify**:

6. ✅ `graflow/coordination/coordinator.py` (base.py)
   - Update `TaskCoordinator.execute_group()` signature to include `handler` parameter

7. ✅ `graflow/coordination/executor.py`
   - Update `GroupExecutor.execute_parallel_group()` signature:
     - Add `handler: Optional[TaskHandler] = None` as 4th parameter
     - Current signature: `execute_parallel_group(group_id, tasks, exec_context)`
     - New signature: `execute_parallel_group(group_id, tasks, exec_context, handler=None)`
   - Simplify handler logic:
     ```python
     if handler is None:
         from graflow.core.handlers.direct import DirectTaskHandler
         handler = DirectTaskHandler()
     ```
   - Update `direct_execute()` signature to accept `handler` parameter
   - Update `_create_coordinator()` to remain unchanged (already takes exec_context)
   - Note: `GroupExecutor.__init__(backend, backend_config)` remains unchanged for backward compatibility

8. ✅ `graflow/coordination/threading.py`
   - Update `ThreadingCoordinator.execute_group()` signature
   - Collect task results into `Dict[str, TaskResult]`
   - Call `handler.on_group_finished()` after all tasks complete
   - Raise `ParallelGroupError` if handler raises it

9. ✅ `graflow/coordination/redis.py`
    - Update `RedisCoordinator.execute_group()` signature
    - Add `_get_completion_results()` method
    - Update `record_task_completion()` to accept `error_message`
    - Call `handler.on_group_finished()` after barrier completes

### Phase 4: Worker Integration (Distributed)

**Files to modify**:

10. ✅ `graflow/worker/worker.py`
    - Update `_process_task_wrapper()` to capture error messages
    - Update `_task_completed()` to pass error messages to queue

11. ✅ `graflow/queue/base.py`
    - Update `TaskQueue.notify_task_completion()` signature to include `error_message`

12. ✅ `graflow/queue/redis.py`
    - Update `RedisTaskQueue.notify_task_completion()` to pass error_message

### Phase 5: Testing

13. ✅ Add unit tests for handlers
14. ✅ Add integration tests for local execution
15. ✅ Add integration tests for distributed execution

### Phase 6: Documentation and Examples

16. ✅ Update CLAUDE.md with new API
17. ✅ Add example files to `examples/` directory:
    - `examples/11_error_handling/parallel_group_strict_mode.py`
      - Default strict mode (all tasks must succeed)
      - Shows ParallelGroupError handling
    - `examples/11_error_handling/parallel_group_best_effort.py`
      - Custom handler: BestEffortHandler (never fail)
      - Demonstrates custom on_group_finished() implementation
    - `examples/11_error_handling/parallel_group_at_least_n.py`
      - Custom handler: AtLeastNSuccessHandler
      - Shows threshold-based success criteria (e.g., 3 out of 4)
    - `examples/11_error_handling/parallel_group_critical_tasks.py`
      - Custom handler: CriticalTasksHandler
      - Shows priority-based success criteria
    - `examples/11_error_handling/parallel_group_composition.py`
      - PolicyDelegatingHandler composition pattern
      - Combines execution handler + policy handler
    - `examples/11_error_handling/README.md`
      - Overview of custom handler patterns
      - When to use each pattern
      - How to implement your own handler
18. ✅ Update README if needed

**Note**: Only `DirectTaskHandler` is built-in to core. All other handlers (BestEffort, AtLeastN, etc.) are user-implemented examples in `examples/` directory. Users copy and adapt these samples for their specific requirements.

---

## Usage Examples

All examples except Example 1 (default) require implementing and registering custom handlers.

### Example 1: Default (Strict Mode)

```python
with workflow("test") as wf:
    @task
    def task_a():
        return "a"

    @task
    def task_b():
        raise Exception("Task B failed!")

    @task
    def task_c():
        return "c"

    # Default: All tasks must succeed
    task_a | task_b | task_c

    engine = WorkflowEngine()
    exec_context = wf.create_execution_context()

    try:
        engine.execute(exec_context)
    except ParallelGroupError as e:
        print(f"❌ Failed: {e.failed_tasks}")
        print(f"✅ Succeeded: {e.successful_tasks}")
        # Output:
        # ❌ Failed: [('task_b', 'Task B failed!')]
        # ✅ Succeeded: ['task_a', 'task_c']
```

### Example 2: Best Effort Mode (Custom Handler)

```python
# First, implement a custom handler
class BestEffortHandler(TaskHandler):
    """Continue even if tasks fail."""

    def get_name(self) -> str:
        return "best_effort"

    def execute_task(self, task, context):
        try:
            result = task.run()
            context.set_result(task.task_id, result)
        except Exception as e:
            context.set_result(task.task_id, e)
            raise

    def on_group_finished(self, group_id, tasks, results, context):
        failed = [tid for tid, r in results.items() if not r.success]
        if failed:
            print(f"⚠️  Group {group_id} completed with {len(failed)} failures (best-effort)")
        # Don't raise exception - always succeed

with workflow("test") as wf:
    @task
    def task_a():
        return "a"

    @task
    def task_b():
        raise Exception("Task B failed!")

    @task
    def task_c():
        return "c"

    # Create and use custom handler instance
    (task_a | task_b | task_c).with_execution(
        handler=BestEffortHandler()
    )

    engine = WorkflowEngine()
    exec_context = wf.create_execution_context()
    engine.execute(exec_context)  # No exception

    # Output:
    # ⚠️  Group completed with 1 failures (best-effort)
```

### Example 3: At Least N Success Handler

```python
class AtLeastNSuccessHandler(TaskHandler):
    """Need at least N tasks to succeed."""

    def __init__(self, min_success: int):
        self.min_success = min_success

    def get_name(self) -> str:
        return f"at_least_{self.min_success}"

    def execute_task(self, task, context):
        try:
            result = task.run()
            context.set_result(task.task_id, result)
        except Exception as e:
            context.set_result(task.task_id, e)
            raise

    def on_group_finished(self, group_id, tasks, results, context):
        successful = [tid for tid, r in results.items() if r.success]

        if len(successful) < self.min_success:
            failed = [(tid, r.error_message) for tid, r in results.items() if not r.success]
            raise ParallelGroupError(
                f"Only {len(successful)}/{self.min_success} tasks succeeded",
                group_id=group_id,
                failed_tasks=failed,
                successful_tasks=successful
            )

# Use: 4 tasks, need at least 3 to succeed
(task_a | task_b | task_c | task_d).with_execution(
    backend=CoordinationBackend.THREADING,
    handler=AtLeastNSuccessHandler(min_success=3)
)
```

### Example 4: Critical Tasks Handler

```python
class CriticalTasksHandler(TaskHandler):
    """Success if critical tasks succeed, ignore others."""

    def __init__(self, critical_task_ids: List[str]):
        self.critical_task_ids = set(critical_task_ids)

    def get_name(self) -> str:
        return "critical_tasks"

    def execute_task(self, task, context):
        try:
            result = task.run()
            context.set_result(task.task_id, result)
        except Exception as e:
            context.set_result(task.task_id, e)
            raise

    def on_group_finished(self, group_id, tasks, results, context):
        failed_critical = [
            tid for tid in self.critical_task_ids
            if tid in results and not results[tid].success
        ]

        if failed_critical:
            raise ParallelGroupError(
                f"Critical tasks failed: {failed_critical}",
                group_id=group_id,
                failed_tasks=[(tid, results[tid].error_message) for tid in failed_critical],
                successful_tasks=[tid for tid, r in results.items() if r.success]
            )

# Use: task_a and task_b are critical, task_c and task_d are optional
(task_a | task_b | task_c | task_d).with_execution(
    handler=CriticalTasksHandler(critical_task_ids=["task_a", "task_b"])
)
```

---

## Backward Compatibility

### Addressing Codex Concern: get_name() Impact on Existing Handlers

**Concern**: Making `get_name()` required breaks all existing TaskHandler implementations.

**Solution**: Provide default implementation in base class.

#### Before (would break existing code):
```python
class TaskHandler(ABC):
    @abstractmethod
    def get_name(self) -> str:  # ← Breaks existing handlers!
        pass
```

#### After (backward compatible):
```python
class TaskHandler(ABC):
    def get_name(self) -> str:
        """Default implementation: Use class name."""
        return self.__class__.__name__  # ← No breaking change!

    @abstractmethod
    def execute_task(...): pass
```

#### Impact Analysis:

| Handler Type | Before | After | Status |
|-------------|--------|-------|--------|
| Existing custom handlers | No `get_name()` | Inherits default | ✅ Works |
| DirectTaskHandler | No `get_name()` | Override to return "direct" | ⚠️ Minor change |
| New custom handlers | - | Can override or use default | ✅ Works |

#### Migration Required:

**Only for built-in handlers** (DirectTaskHandler, DockerTaskHandler, etc.):

```python
# graflow/core/handlers/direct.py (ONE TIME CHANGE)
class DirectTaskHandler(TaskHandler):
    def get_name(self) -> str:
        return "direct"  # ← Add this method

    def execute_task(self, task, context):
        # ... existing implementation unchanged
```

**User custom handlers**: No changes required - they inherit `__class__.__name__`.

### Breaking Changes

⚠️ **Behavior Change**: The default group execution will change from "ignore failures" to "fail on any failure".

**Before** (current broken behavior):
```python
(task_a | task_b | task_c)  # If task_b fails, workflow continues (BUG!)
```

**After** (new correct behavior):
```python
(task_a | task_b | task_c)  # If task_b fails, raises ParallelGroupError
```

This is intentional and necessary to fix the broken behavior. Users should not rely on the old behavior.

### Migration Path

If users need failure-tolerant behavior, implement a custom handler:

```python
# Implement best-effort handler
class BestEffortHandler(TaskHandler):
    def get_name(self) -> str:
        return "best_effort"  # Optional override

    def execute_task(self, task, context):
        # Direct execution
        try:
            result = task.run()
            context.set_result(task.task_id, result)
        except Exception as e:
            context.set_result(task.task_id, e)
            raise

    def on_group_finished(self, group_id, tasks, results, context):
        # Don't raise exception - always succeed
        failed = [tid for tid, r in results.items() if not r.success]
        if failed:
            print(f"⚠️  {len(failed)} tasks failed (best-effort mode)")

# Register and use
engine.register_handler("best_effort", BestEffortHandler())
(task_a | task_b | task_c).with_execution(handler="best_effort")
```

**Rationale**: The current behavior is a bug. Workflows should not silently continue when tasks fail.

---

## Future Extensions

### 1. Retry Support

```python
class RetryOnFailureHandler(TaskHandler):
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def on_group_finished(self, group_id, tasks, results, context):
        failed_tasks = [tid for tid, r in results.items() if not r.success]

        if failed_tasks and context.retry_count < self.max_retries:
            # Re-queue failed tasks
            for task_id in failed_tasks:
                task = context.graph.get_node(task_id)
                context.add_to_queue(task)
            context.retry_count += 1
        elif failed_tasks:
            raise ParallelGroupError(...)
```

### 2. Conditional Handlers

```python
class ConditionalHandler(TaskHandler):
    def __init__(self, condition: Callable[[Dict[str, TaskResult]], bool]):
        self.condition = condition

    def on_group_finished(self, group_id, tasks, results, context):
        if not self.condition(results):
            raise ParallelGroupError(...)

# Usage
engine.register_handler(
    "custom_condition",
    ConditionalHandler(lambda results: results["task_a"].success or results["task_b"].success)
)
```

### 3. Weighted Success

```python
class WeightedSuccessHandler(TaskHandler):
    def __init__(self, weights: Dict[str, float], threshold: float = 0.75):
        self.weights = weights
        self.threshold = threshold

    def on_group_finished(self, group_id, tasks, results, context):
        total_weight = sum(self.weights.values())
        success_weight = sum(
            self.weights.get(tid, 1.0)
            for tid, result in results.items()
            if result.success
        )

        if success_weight / total_weight < self.threshold:
            raise ParallelGroupError(...)
```

---

## References

- Existing design: `docs/with_execution_implementation_design.md`
- Test pattern: `tests/scenario/test_custom_handler_workflow.py`
- Related: `docs/parallel_execution_queue_design.md`
