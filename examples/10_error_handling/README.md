# Parallel Group Error Handling Examples

This directory contains comprehensive examples of parallel group error handling in Graflow. These examples demonstrate different error handling strategies for parallel task execution.

## Overview

When executing tasks in parallel, not all failures are equal. Graflow provides a flexible handler system with built-in policies and extensibility:

- **Strict Mode** (default via :class:`StrictGroupPolicy`): All tasks must succeed
- **Best-effort** (:class:`BestEffortGroupPolicy`): Continue even if some tasks fail
- **At-least-N** (:class:`AtLeastNGroupPolicy`): Require a minimum number of successes
- **Critical Tasks** (:class:`CriticalGroupPolicy`): Only fail if specific tasks fail

You can also implement your own policy by subclassing :class:`graflow.core.handlers.GroupExecutionPolicy`
and overriding :meth:`on_group_finished` for custom behavior. When passing a policy instance to
``with_execution(...)`` it is automatically adapted to the task handler interface.

## Quick Start

Run any example:

```bash
PYTHONPATH=. uv run python examples/11_error_handling/parallel_group_strict_mode.py
PYTHONPATH=. uv run python examples/11_error_handling/parallel_group_best_effort.py
PYTHONPATH=. uv run python examples/11_error_handling/parallel_group_at_least_n.py
PYTHONPATH=. uv run python examples/11_error_handling/parallel_group_critical_tasks.py
PYTHONPATH=. uv run python examples/11_error_handling/parallel_group_custom_policy.py
```

Or use the make command:

```bash
make py examples/11_error_handling/parallel_group_strict_mode.py
```

## Examples

### 1. Strict Mode (`parallel_group_strict_mode.py`)

**Default behavior** - all tasks must succeed or the workflow fails.

```python
from graflow.exceptions import ParallelGroupError
from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.decorators import task
from graflow.core.workflow import workflow

with workflow("strict_mode") as wf:
    @task
    def task_a():
        return "success"

    @task
    def task_b():
        raise ValueError("Task failed!")

    # Default strict mode - any failure raises ParallelGroupError
    parallel = (task_a | task_b).with_execution(
        backend=CoordinationBackend.THREADING
    )

    try:
        wf.execute()
    except ParallelGroupError as e:
        print(f"Failed tasks: {e.failed_tasks}")
        print(f"Successful tasks: {e.successful_tasks}")
```

**Use cases:**
- Critical operations where all tasks must succeed
- Data validation pipelines
- Financial transactions
- Security-sensitive operations

**Key concepts:**
- `ParallelGroupError` contains detailed failure information
- Lists both failed and successful tasks
- Works with all coordination backends (DIRECT, THREADING, REDIS)

### 2. Best-effort Mode (`parallel_group_best_effort.py`)

**Never fails** - continues even when tasks fail. Useful for non-critical operations.

```python
from graflow.core.handlers.group_policy import BestEffortGroupPolicy

with workflow("best_effort") as wf:
    # ... define tasks ...

    parallel = (task_a | task_b | task_c).with_execution(
        backend=CoordinationBackend.THREADING,
        policy="best_effort"
    )

    wf.execute()  # Won't raise exception even if tasks fail
```

**Use cases:**
- Multi-channel notifications (email, SMS, Slack)
- Data enrichment from multiple sources
- Parallel exports to different formats
- Optional analytics or logging

**Key concepts:**
- Built-in handler logs failures (via `logging`) but never raises
- Handler can inspect all task results when you need custom logic
- Workflow continues to successor tasks

### 3. At-least-N Success (`parallel_group_at_least_n.py`)

**Quorum-based** - requires a minimum number of tasks to succeed.

```python
with workflow("quorum") as wf:
    # ... define tasks ...

    # Require at least 2 out of 4 to succeed
    from graflow.core.handlers.group_policy import AtLeastNGroupPolicy

    parallel = (task_a | task_b | task_c | task_d).with_execution(
        backend=CoordinationBackend.THREADING,
        policy=AtLeastNGroupPolicy(min_success=2)
    )

    wf.execute()
```

**Percentage-based variant:**

```python
class PercentageSuccessHandler(DirectTaskHandler):
    def __init__(self, min_percentage: float):
        self.min_percentage = min_percentage

    def on_group_finished(self, group_id, tasks, results, context):
        success_rate = sum(1 for r in results.values() if r.success) / len(results)

        if success_rate < self.min_percentage:
            # Raise error with detailed info
            ...

# Require 70% success rate
handler = PercentageSuccessHandler(min_percentage=0.70)
```

**Use cases:**
- Multi-region deployments (need quorum)
- Redundant data sources
- Distributed consensus
- High-availability systems

**Key concepts:**
- Configurable success thresholds with built-in handler
- Percentage-based custom policies are easy to implement
- Detailed error reporting when threshold not met

### 4. Critical Tasks (`parallel_group_critical_tasks.py`)

**Priority-based** - only fail if critical tasks fail. Optional tasks can fail.

```python
from graflow.core.handlers.group_policy import CriticalGroupPolicy

with workflow("critical") as wf:
    @task
    def extract_data():  # Critical
        return load_data()

    @task
    def validate_schema():  # Critical
        return validate()

    @task
    def enrich_metadata():  # Optional
        return enrich()  # Can fail

    # Only extract_data and validate_schema are critical
    parallel = (extract_data | validate_schema | enrich_metadata).with_execution(
        backend=CoordinationBackend.THREADING,
        policy=CriticalGroupPolicy(
            critical_task_ids=["extract_data", "validate_schema"]
        )
    )

    wf.execute()
```

**Use cases:**
- Data pipelines with required + optional steps
- Notifications with required channels + nice-to-have
- ML training with core + analysis tasks
- Multi-stage deployments

**Key concepts:**
- Separate critical from optional tasks
- Built-in policy raises if any critical task fails
- Optional failures are logged (via `logging`) but don't stop execution

### 5. Custom Policy (`parallel_group_custom_policy.py`)

**Domain-specific rules** - create a policy tailored to your workflow.

```python
from graflow.core.handlers.group_policy import GroupExecutionPolicy
from graflow.exceptions import ParallelGroupError


class CriticalLimitedFailuresPolicy(GroupExecutionPolicy):
    def __init__(self, critical_task_ids: list[str], max_failures: int):
        self.critical_task_ids = critical_task_ids
        self.max_failures = max_failures

    def on_group_finished(self, group_id, tasks, results, context):
        self._validate_group_results(group_id, tasks, results)

        failed = [(tid, r.error_message) for tid, r in results.items() if not r.success]
        failed_critical = [tid for tid in self.critical_task_ids if not results[tid].success]

        if failed_critical:
            raise ParallelGroupError(
                f"Critical failed: {failed_critical}",
                group_id=group_id,
                failed_tasks=[(tid, results[tid].error_message) for tid in failed_critical],
                successful_tasks=[tid for tid, r in results.items() if r.success],
            )

        if len(failed) > self.max_failures:
            raise ParallelGroupError(
                f"Exceeded failure budget {self.max_failures}",
                group_id=group_id,
                failed_tasks=failed,
                successful_tasks=[tid for tid, r in results.items() if r.success],
            )


parallel = (task_a | task_b | task_c).with_execution(
    policy=CriticalLimitedFailuresPolicy(["task_a"], max_failures=1)
)
```

**Use cases:**
- Complex workflows with both critical and optional steps
- Gradual rollouts where a limited number of failures is acceptable
- Compliance scenarios requiring explicit auditing of failures

**Key concepts:**
- Custom policies encapsulate domain logic
- Always call ``_validate_group_results`` to ensure result integrity
- Raise ``ParallelGroupError`` to propagate a controlled failure

## Policy Architecture

Custom group policies extend `GroupExecutionPolicy`:

```python
from graflow.core.handlers.group_policy import GroupExecutionPolicy
from graflow.exceptions import ParallelGroupError


class MyCustomPolicy(GroupExecutionPolicy):
    def on_group_finished(self, group_id, tasks, results, context):
        self._validate_group_results(group_id, tasks, results)

        failed = [(tid, r.error_message) for tid, r in results.items() if not r.success]
        if failed:
            raise ParallelGroupError(
                "Custom policy failure",
                group_id=group_id,
                failed_tasks=failed,
                successful_tasks=[tid for tid, r in results.items() if r.success],
            )


# Usage
parallel = (task_a | task_b).with_execution(policy=MyCustomPolicy())

# Execute
try:
    results = parallel.run()
    print("Policy succeeded:", results)
except ParallelGroupError as exc:
    print("Policy failed:", exc.failed_tasks)
```

### TaskResult Structure

```python
@dataclass
class TaskResult:
    task_id: str
    success: bool                    # True if task completed without error
    error_message: Optional[str]     # Error message if failed
    duration: float                  # Execution time in seconds
    timestamp: float                 # Completion timestamp
```

### ParallelGroupError

```python
class ParallelGroupError(GraflowRuntimeError):
    def __init__(
        self,
        message: str,
        group_id: str,
        failed_tasks: list[tuple[str, str]],      # [(task_id, error_msg), ...]
        successful_tasks: list[str]                # [task_id, ...]
    ):
        ...
```

## Choosing the Right Policy

| Policy | Use When | Example |
|--------|----------|---------|
| **Strict Mode** (default) | All tasks must succeed | Critical operations, financial transactions |
| **Best-effort** | Partial success acceptable | Notifications, analytics, optional enrichment |
| **At-least-N** | Need quorum/redundancy | Multi-region deploys, data source redundancy |
| **Critical Tasks** | Mixed priority tasks | Pipelines with required + optional steps |

## Advanced Patterns

### Handler Composition

You can combine multiple policies:

```python
class CompositeHandler(DirectTaskHandler):
    def __init__(self, critical_tasks, min_success):
        self.critical_tasks = set(critical_tasks)
        self.min_success = min_success

    def on_group_finished(self, group_id, tasks, results, context):
        # First check critical tasks
        failed_critical = [tid for tid in self.critical_tasks if not results[tid].success]
        if failed_critical:
            raise ParallelGroupError("Critical tasks failed", ...)

        # Then check minimum success
        success_count = sum(1 for r in results.values() if r.success)
        if success_count < self.min_success:
            raise ParallelGroupError("Minimum success not met", ...)
```

### Context-aware Handlers

Access workflow state to make dynamic decisions:

```python
class ContextAwareHandler(DirectTaskHandler):
    def on_group_finished(self, group_id, tasks, results, context):
        # Check workflow state
        previous_failures = context.metadata.get("failure_count", 0)

        # Adjust policy based on context
        if previous_failures > 0:
            # Be more lenient after previous failures
            threshold = 0.5
        else:
            # Be strict on first try
            threshold = 1.0

        success_rate = sum(1 for r in results.values() if r.success) / len(results)
        if success_rate < threshold:
            raise ParallelGroupError(...)
```

### Retry Logic

Handlers can trigger retries:

```python
class RetryHandler(DirectTaskHandler):
    def on_group_finished(self, group_id, tasks, results, context):
        failed = [t for t in tasks if not results[t.task_id].success]

        if failed:
            retry_count = context.metadata.get(f"{group_id}_retries", 0)

            if retry_count < 3:
                # Store retry count
                context.metadata[f"{group_id}_retries"] = retry_count + 1

                # Re-submit failed tasks
                for task in failed:
                    context.next_task(task)
            else:
                raise ParallelGroupError("Max retries exceeded", ...)
```

## Testing

All handlers include comprehensive test coverage:

- **Unit tests**: `tests/core/test_parallel_group_error_handling.py`
  - Test handler behavior in isolation
  - 13 tests covering all handler types

- **Scenario tests**: `tests/scenario/test_parallel_group_error_handling_scenarios.py`
  - Test handlers in real workflow execution
  - 11 tests covering integration scenarios

Run tests:

```bash
# Unit tests
PYTHONPATH=. uv run pytest tests/core/test_parallel_group_error_handling.py -v

# Scenario tests
PYTHONPATH=. uv run pytest tests/scenario/test_parallel_group_error_handling_scenarios.py -v

# All error handling tests
PYTHONPATH=. uv run pytest tests/ -k "parallel_group_error" -v
```

## Best Practices

1. **Choose the right policy**: Match handler to business requirements
2. **Provide clear error messages**: Help debugging with detailed ParallelGroupError messages
3. **Log appropriately**: Use print/logging in handlers for visibility
4. **Test failure scenarios**: Verify handler behavior when tasks fail
5. **Document critical tasks**: Clearly identify which tasks are critical
6. **Consider retry logic**: Some failures may be transient
7. **Monitor metrics**: Track success rates, failure patterns
8. **Use handler composition**: Combine policies for complex requirements

## Performance Considerations

- Handlers execute after all tasks complete (no overhead during execution)
- Handler logic should be fast (avoid expensive operations)
- Results are already collected - no additional task inspection needed
- Handlers are serializable for distributed execution

## Migration Guide

### From Previous Versions

If you were catching exceptions manually:

**Before:**
```python
try:
    parallel = (task_a | task_b | task_c).run()
except Exception as e:
    # Manual error handling
    ...
```

**After:**
```python
parallel = (task_a | task_b | task_c).with_execution(
    handler=YourCustomHandler()
)
parallel.run()  # Handler automatically manages errors
```

### From Strict Mode to Custom Handler

**Before (implicit strict mode):**
```python
parallel = (task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.THREADING
)
```

**After (explicit custom handler):**
```python
from graflow.core.handlers import BestEffortGroupPolicy

from graflow.core.handlers import BestEffortGroupPolicy

parallel = (task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.THREADING,
    handler=BestEffortGroupPolicy()
)
```

## Related Documentation

- Design document: `docs/parallel_group_error_handling_design.md`
- Core handler implementation: `graflow/core/handler.py`
- DirectTaskHandler: `graflow/core/handlers/direct.py`
- Exceptions: `graflow/exceptions.py`

## Support

For questions or issues:
1. Check the examples in this directory
2. Review unit and scenario tests
3. Consult the design document
4. File an issue on GitHub
