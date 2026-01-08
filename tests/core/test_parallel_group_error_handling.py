"""Unit tests for parallel group error handling with TaskHandler."""

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.handler import TaskResult
from graflow.core.handlers.direct import DirectTaskHandler
from graflow.core.handlers.group_policy import (
    AtLeastNGroupPolicy,
    BestEffortGroupPolicy,
    CriticalGroupPolicy,
    GroupExecutionPolicy,
    StrictGroupPolicy,
    canonicalize_group_policy,
    resolve_group_policy,
)
from graflow.core.workflow import workflow
from graflow.exceptions import ParallelGroupError


class TestTaskResult:
    """Test TaskResult dataclass."""

    def test_task_result_creation(self):
        """Test creating a TaskResult."""
        result = TaskResult(task_id="test_task", success=True, error_message=None, duration=1.5, timestamp=1234567890.0)

        assert result.task_id == "test_task"
        assert result.success is True
        assert result.error_message is None
        assert result.duration == 1.5
        assert result.timestamp == 1234567890.0

    def test_task_result_failure(self):
        """Test creating a TaskResult for a failed task."""
        result = TaskResult(
            task_id="failed_task",
            success=False,
            error_message="Task failed: division by zero",
            duration=0.5,
            timestamp=1234567890.0,
        )

        assert result.task_id == "failed_task"
        assert result.success is False
        assert result.error_message == "Task failed: division by zero"


class TestParallelGroupError:
    """Test ParallelGroupError exception."""

    def test_parallel_group_error_creation(self):
        """Test creating a ParallelGroupError."""
        error = ParallelGroupError(
            message="Parallel group failed",
            group_id="group_1",
            failed_tasks=[("task_a", "Error message")],
            successful_tasks=["task_b", "task_c"],
        )

        assert str(error) == "Parallel group failed"
        assert error.group_id == "group_1"
        assert error.failed_tasks == [("task_a", "Error message")]
        assert error.successful_tasks == ["task_b", "task_c"]

    def test_parallel_group_error_is_runtime_error(self):
        """Test that ParallelGroupError is a subclass of GraflowRuntimeError."""
        from graflow.exceptions import GraflowRuntimeError

        error = ParallelGroupError(message="Test error", group_id="group_1", failed_tasks=[], successful_tasks=[])

        assert isinstance(error, GraflowRuntimeError)


class TestDirectTaskHandler:
    """Test DirectTaskHandler."""

    def test_get_name(self):
        """Test DirectTaskHandler.get_name()."""
        handler = DirectTaskHandler()
        assert handler.get_name() == "direct"


class TestPolicyHandlers:
    """Test built-in parallel group policy handlers."""

    def test_strict_group_handler_matches_default(self):
        """Strict handler should raise when any task fails."""
        handler = StrictGroupPolicy()

        with workflow("test") as wf:

            @task
            def task_a():
                return "a"

            @task
            def task_b():
                raise Exception("Task B failed!")

            tasks = [task_a, task_b]
            results = {
                "task_a": TaskResult(task_id="task_a", success=True),
                "task_b": TaskResult(task_id="task_b", success=False, error_message="Task B failed!"),
            }

            with pytest.raises(ParallelGroupError):
                handler.on_group_finished(
                    "group_1", tasks, results, ExecutionContext.create(wf.graph, start_node="task_a")
                )

    def test_best_effort_handler(self):
        """Best-effort handler should not raise when tasks fail."""
        handler = BestEffortGroupPolicy()

        with workflow("test") as wf:

            @task
            def task_a():
                return "a"

            @task
            def task_b():
                raise Exception("Task B failed!")

            tasks = [task_a, task_b]
            results = {
                "task_a": TaskResult(task_id="task_a", success=True),
                "task_b": TaskResult(task_id="task_b", success=False, error_message="Task B failed!"),
            }

            handler.on_group_finished("group_1", tasks, results, ExecutionContext.create(wf.graph, start_node="task_a"))

    def test_at_least_n_success_handler(self):
        """At-least-N handler should enforce minimum success count."""

        handler = AtLeastNGroupPolicy(min_success=2)

        with workflow("test") as wf:

            @task
            def task_a():
                return "a"

            @task
            def task_b():
                return "b"

            @task
            def task_c():
                return "c"

            @task
            def task_d():
                raise Exception("Task D failed!")

            tasks = [task_a, task_b, task_c, task_d]
            results = {
                "task_a": TaskResult(task_id="task_a", success=True),
                "task_b": TaskResult(task_id="task_b", success=True),
                "task_c": TaskResult(task_id="task_c", success=True),
                "task_d": TaskResult(task_id="task_d", success=False, error_message="Task D failed!"),
            }

            handler.on_group_finished("group_1", tasks, results, ExecutionContext.create(wf.graph, start_node="task_a"))

        handler2 = AtLeastNGroupPolicy(min_success=2)

        with workflow("test2") as wf2:

            @task
            def task_e():
                return "e"

            @task
            def task_f():
                raise Exception("Task F failed!")

            @task
            def task_g():
                raise Exception("Task G failed!")

            @task
            def task_h():
                raise Exception("Task H failed!")

            tasks2 = [task_e, task_f, task_g, task_h]
            results2 = {
                "task_e": TaskResult(task_id="task_e", success=True),
                "task_f": TaskResult(task_id="task_f", success=False, error_message="Task F failed!"),
                "task_g": TaskResult(task_id="task_g", success=False, error_message="Task G failed!"),
                "task_h": TaskResult(task_id="task_h", success=False, error_message="Task H failed!"),
            }

            with pytest.raises(ParallelGroupError) as exc_info:
                handler2.on_group_finished(
                    "group_2", tasks2, results2, ExecutionContext.create(wf2.graph, start_node="task_e")
                )

            error = exc_info.value
            assert "1/2" in str(error)
            assert len(error.failed_tasks) == 3
            assert len(error.successful_tasks) == 1

    def test_critical_tasks_handler(self):
        """Critical handler should fail only when critical tasks fail."""
        handler = CriticalGroupPolicy(critical_task_ids=["critical_task"])

        with workflow("test") as wf:

            @task
            def critical_task():
                return "critical success"

            @task
            def optional_task():
                raise Exception("Optional task failed!")

            tasks = [critical_task, optional_task]
            results = {
                "critical_task": TaskResult(task_id="critical_task", success=True),
                "optional_task": TaskResult(
                    task_id="optional_task", success=False, error_message="Optional task failed!"
                ),
            }

            handler.on_group_finished(
                "group_1", tasks, results, ExecutionContext.create(wf.graph, start_node="critical_task")
            )

        handler_failure = CriticalGroupPolicy(critical_task_ids=["critical_task_fail"])

        with workflow("test_failure") as wf_failure:

            @task
            def critical_task_fail():
                raise Exception("Critical task failed!")

            @task
            def optional_task_success():
                return "optional"

            tasks_failure = [critical_task_fail, optional_task_success]
            results_failure = {
                "critical_task_fail": TaskResult(
                    task_id="critical_task_fail", success=False, error_message="Critical task failed!"
                ),
                "optional_task_success": TaskResult(task_id="optional_task_success", success=True),
            }

            with pytest.raises(ParallelGroupError) as exc_info:
                handler_failure.on_group_finished(
                    "group_critical",
                    tasks_failure,
                    results_failure,
                    ExecutionContext.create(wf_failure.graph, start_node="critical_task_fail"),
                )

            assert "critical_task_fail" in str(exc_info.value)


class TestHandlerInheritance:
    """Test handler inheritance patterns."""

    def test_custom_handler_inherits_execute_task(self):
        """Test that custom handler inherits execute_task from DirectTaskHandler."""

        class MyPolicyHandler(DirectTaskHandler):
            """Custom policy handler."""

            def get_name(self):
                return "my_policy"

            def on_group_finished(self, group_id, tasks, results, context):
                # Custom logic
                pass

        handler = MyPolicyHandler()

        # Should have execute_task method from DirectTaskHandler
        assert hasattr(handler, "execute_task")
        assert callable(handler.execute_task)

        # Should have get_name method
        assert handler.get_name() == "my_policy"

    def test_default_get_name_implementation(self):
        """Test that default get_name() returns class name."""

        class MyCustomHandler(DirectTaskHandler):
            """Custom handler without get_name override."""

            def on_group_finished(self, group_id, tasks, results, context):
                pass

        handler = MyCustomHandler()

        # Default implementation should return class name
        assert handler.get_name() == "direct"  # Inherits from DirectTaskHandler


class TestPolicySerialization:
    """Validate serialization helpers for group execution policies."""

    def test_serialize_and_resolve_strict_policy(self):
        """String policies round-trip without instantiation issues."""
        serialized = canonicalize_group_policy("strict")
        assert serialized == "strict"

        resolved = resolve_group_policy(serialized)
        assert isinstance(resolved, StrictGroupPolicy)

    def test_serialize_and_resolve_best_effort_policy(self):
        serialized = canonicalize_group_policy(BestEffortGroupPolicy())
        assert serialized == "best_effort"
        resolved = resolve_group_policy(serialized)
        assert isinstance(resolved, BestEffortGroupPolicy)

    def test_serialize_and_resolve_critical_policy(self):
        policy = CriticalGroupPolicy(["task_a", "task_b"])
        serialized = canonicalize_group_policy(policy)

        assert serialized == {
            "type": "critical",
            "critical_task_ids": ["task_a", "task_b"],
        }

        resolved = resolve_group_policy(serialized)
        assert isinstance(resolved, CriticalGroupPolicy)
        assert resolved.critical_task_ids == ["task_a", "task_b"]

    def test_serialize_and_resolve_at_least_n_policy(self):
        policy = AtLeastNGroupPolicy(min_success=3)
        serialized = canonicalize_group_policy(policy)

        assert serialized == {"type": "at_least_n", "min_success": 3}

        resolved = resolve_group_policy(serialized)
        assert isinstance(resolved, AtLeastNGroupPolicy)
        assert resolved.min_success == 3

    def test_resolve_raises_for_unknown_policy(self):
        with pytest.raises(ValueError):
            resolve_group_policy("unknown_policy")

        with pytest.raises(ValueError):
            resolve_group_policy({"type": "unknown"})

    def test_canonicalize_and_resolve_custom_policy(self):
        class DummyPolicy(GroupExecutionPolicy):
            def on_group_finished(self, group_id, tasks, results, context):
                pass

        custom_policy = DummyPolicy()

        serialized = canonicalize_group_policy(custom_policy)
        assert serialized == {"type": "__custom__", "policy": custom_policy}

        resolved = resolve_group_policy(serialized)
        assert resolved is custom_policy
