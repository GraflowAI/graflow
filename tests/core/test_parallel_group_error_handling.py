"""Unit tests for parallel group error handling with TaskHandler."""

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.handler import TaskResult
from graflow.core.handlers.direct import DirectTaskHandler
from graflow.core.workflow import workflow
from graflow.exceptions import ParallelGroupError


class TestTaskResult:
    """Test TaskResult dataclass."""

    def test_task_result_creation(self):
        """Test creating a TaskResult."""
        result = TaskResult(
            task_id="test_task",
            success=True,
            error_message=None,
            duration=1.5,
            timestamp=1234567890.0
        )

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
            timestamp=1234567890.0
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
            successful_tasks=["task_b", "task_c"]
        )

        assert str(error) == "Parallel group failed"
        assert error.group_id == "group_1"
        assert error.failed_tasks == [("task_a", "Error message")]
        assert error.successful_tasks == ["task_b", "task_c"]

    def test_parallel_group_error_is_runtime_error(self):
        """Test that ParallelGroupError is a subclass of GraflowRuntimeError."""
        from graflow.exceptions import GraflowRuntimeError

        error = ParallelGroupError(
            message="Test error",
            group_id="group_1",
            failed_tasks=[],
            successful_tasks=[]
        )

        assert isinstance(error, GraflowRuntimeError)


class TestDirectTaskHandler:
    """Test DirectTaskHandler with on_group_finished."""

    def test_get_name(self):
        """Test DirectTaskHandler.get_name()."""
        handler = DirectTaskHandler()
        assert handler.get_name() == "direct"

    def test_on_group_finished_all_success(self):
        """Test on_group_finished with all tasks successful."""
        handler = DirectTaskHandler()

        with workflow("test") as wf:
            @task
            def task_a():
                return "a"

            @task
            def task_b():
                return "b"

            tasks = [task_a, task_b]
            results = {
                "task_a": TaskResult(task_id="task_a", success=True),
                "task_b": TaskResult(task_id="task_b", success=True)
            }

            exec_context = ExecutionContext.create(wf.graph, "test")

            # Should not raise exception
            handler.on_group_finished("group_1", tasks, results, exec_context)

    def test_on_group_finished_one_failure(self):
        """Test on_group_finished with one task failure."""
        handler = DirectTaskHandler()

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
                "task_b": TaskResult(task_id="task_b", success=False, error_message="Task B failed!")
            }

            # Should raise ParallelGroupError
            with pytest.raises(ParallelGroupError) as exc_info:
                handler.on_group_finished("group_1", tasks, results, ExecutionContext.create(wf.graph, "test"))

            error = exc_info.value
            assert error.group_id == "group_1"
            assert len(error.failed_tasks) == 1
            assert error.failed_tasks[0] == ("task_b", "Task B failed!")
            assert error.successful_tasks == ["task_a"]

    def test_on_group_finished_all_failures(self):
        """Test on_group_finished with all tasks failing."""
        handler = DirectTaskHandler()

        with workflow("test") as wf:
            @task
            def task_a():
                raise Exception("Task A failed!")

            @task
            def task_b():
                raise Exception("Task B failed!")

            tasks = [task_a, task_b]
            results = {
                "task_a": TaskResult(task_id="task_a", success=False, error_message="Task A failed!"),
                "task_b": TaskResult(task_id="task_b", success=False, error_message="Task B failed!")
            }

            # Should raise ParallelGroupError
            with pytest.raises(ParallelGroupError) as exc_info:
                handler.on_group_finished("group_1", tasks, results, ExecutionContext.create(wf.graph, "test"))

            error = exc_info.value
            assert error.group_id == "group_1"
            assert len(error.failed_tasks) == 2
            assert error.successful_tasks == []

    def test_on_group_finished_missing_results(self):
        """Test on_group_finished with missing task results."""
        handler = DirectTaskHandler()

        with workflow("test") as wf:
            @task
            def task_a():
                return "a"

            @task
            def task_b():
                return "b"

            tasks = [task_a, task_b]
            # Missing task_b result
            results = {
                "task_a": TaskResult(task_id="task_a", success=True),
            }

            # Should raise ParallelGroupError for missing results
            with pytest.raises(ParallelGroupError) as exc_info:
                handler.on_group_finished("group_1", tasks, results, ExecutionContext.create(wf.graph, "test"))

            error = exc_info.value
            assert "missing results" in str(error).lower()


class TestCustomHandlers:
    """Test custom handler implementations."""

    def test_best_effort_handler(self):
        """Test best-effort handler that never fails."""

        class BestEffortHandler(DirectTaskHandler):
            """Continue even if tasks fail."""

            def get_name(self):
                return "best_effort"

            def on_group_finished(self, group_id, tasks, results, context):  # type: ignore[override]
                # Never raise exception - always succeed
                failed = [tid for tid, r in results.items() if not r.success]
                if failed:
                    print(f"⚠️  Group {group_id} completed with {len(failed)} failures (best-effort)")

        handler = BestEffortHandler()

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
                "task_b": TaskResult(task_id="task_b", success=False, error_message="Task B failed!")
            }

            # Should NOT raise exception
            handler.on_group_finished("group_1", tasks, results, ExecutionContext.create(wf.graph, "test"))

    def test_at_least_n_success_handler(self):
        """Test at-least-N-success handler."""

        class AtLeastNSuccessHandler(DirectTaskHandler):
            """Require at least N tasks to succeed."""

            def __init__(self, min_success: int):
                self.min_success = min_success

            def get_name(self):
                return f"at_least_{self.min_success}"

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

        # Test case 1: 3 out of 4 succeed (min_success=2) - should pass
        handler = AtLeastNSuccessHandler(min_success=2)

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
                "task_d": TaskResult(task_id="task_d", success=False, error_message="Task D failed!")
            }

            # Should NOT raise exception (3 >= 2)
            handler.on_group_finished("group_1", tasks, results, ExecutionContext.create(wf.graph, "test"))

        # Test case 2: 1 out of 4 succeed (min_success=2) - should fail
        handler2 = AtLeastNSuccessHandler(min_success=2)

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
                "task_h": TaskResult(task_id="task_h", success=False, error_message="Task H failed!")
            }

            # Should raise ParallelGroupError (1 < 2)
            with pytest.raises(ParallelGroupError) as exc_info:
                handler2.on_group_finished("group_2", tasks2, results2, ExecutionContext.create(wf2.graph, "test2"))

            error = exc_info.value
            assert "1/2" in str(error)
            assert len(error.failed_tasks) == 3
            assert len(error.successful_tasks) == 1


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
