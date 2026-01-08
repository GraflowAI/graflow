"""Tests for ParallelGroup.with_execution() method."""

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.handlers.group_policy import AtLeastNGroupPolicy
from graflow.core.task import ParallelGroup, Task
from graflow.core.workflow import WorkflowContext


class TestWithExecution:
    """Test ParallelGroup.with_execution() method."""

    def test_with_execution_sets_backend(self):
        """Test that with_execution() sets the backend."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            result = group.with_execution(backend=CoordinationBackend.THREADING)

            assert group._execution_config["backend"] == CoordinationBackend.THREADING
            assert result is group  # Method chaining

    def test_with_execution_sets_backend_config(self):
        """Test that with_execution() sets backend_config."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            group.with_execution(backend_config={"thread_count": 4})

            assert group._execution_config["backend_config"]["thread_count"] == 4

    def test_with_execution_merges_backend_config(self):
        """Test that backend_config is merged on multiple calls."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            group.with_execution(backend_config={"thread_count": 4})
            group.with_execution(backend_config={"timeout": 30})

            config = group._execution_config["backend_config"]
            assert config["thread_count"] == 4
            assert config["timeout"] == 30

    def test_with_execution_stores_policy_string(self):
        """Passing a built-in policy name stores the serialized string."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            group.with_execution(policy="best_effort")

            assert group._execution_config["policy"] == "best_effort"

    def test_with_execution_serializes_policy_instance(self):
        """Passing a policy instance stores a lightweight configuration."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            group.with_execution(policy=AtLeastNGroupPolicy(min_success=2))

            assert group._execution_config["policy"] == {"type": "at_least_n", "min_success": 2}

    def test_with_execution_method_chaining(self):
        """Test method chaining with with_execution()."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = task1 | task2

            result = group.with_execution(backend=CoordinationBackend.REDIS).set_group_name("my_group")

            assert isinstance(result, ParallelGroup)
            assert result.task_id == "my_group"
            assert result._execution_config["backend"] == CoordinationBackend.REDIS

    def test_with_execution_both_params(self):
        """Test with_execution() with both backend and backend_config."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            group.with_execution(backend=CoordinationBackend.THREADING, backend_config={"thread_count": 2})

            assert group._execution_config["backend"] == CoordinationBackend.THREADING
            assert group._execution_config["backend_config"]["thread_count"] == 2

    def test_default_execution_config(self):
        """Test that _execution_config is initialized with None backend."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            assert group._execution_config["backend"] is None
            assert group._execution_config["backend_config"] == {}


class TestWithExecutionRunBehavior:
    """Test run() method behavior with with_execution()."""

    def test_run_calls_static_executor_with_configured_backend(self, mocker):
        """Test that run() calls GroupExecutor.execute_parallel_group() with configured backend."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            group.with_execution(backend=CoordinationBackend.THREADING)

            graph = TaskGraph()
            context = ExecutionContext(graph)
            group.set_execution_context(context)

            # Mock the static GroupExecutor.execute_parallel_group method
            mock_execute = mocker.patch("graflow.core.task.GroupExecutor.execute_parallel_group")

            group.run()

            # Verify static method was called with correct arguments
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[0][0] == group.task_id  # group_id
            assert call_args[0][1] == group.tasks  # tasks
            assert call_args[0][2] == context  # context
            assert call_args[1]["backend"] == CoordinationBackend.THREADING
            assert call_args[1]["policy"] == group._execution_config["policy"]

    def test_run_calls_static_executor_with_default_backend(self, mocker):
        """Test that run() calls GroupExecutor.execute_parallel_group() with default backend when not configured."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            # Don't call with_execution() - backend remains None
            assert group._execution_config["backend"] is None

            graph = TaskGraph()
            context = ExecutionContext(graph)
            group.set_execution_context(context)

            # Mock the static GroupExecutor.execute_parallel_group method
            mock_execute = mocker.patch("graflow.core.task.GroupExecutor.execute_parallel_group")

            group.run()

            # Verify static method was called with None backend (will use default)
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[1]["backend"] is None  # None means use default
            assert call_args[1]["policy"] == group._execution_config["policy"]

    def test_run_passes_backend_config(self, mocker):
        """Test that run() passes backend_config to GroupExecutor."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            group.with_execution(backend=CoordinationBackend.THREADING, backend_config={"thread_count": 4})

            graph = TaskGraph()
            context = ExecutionContext(graph)
            group.set_execution_context(context)

            # Mock the static GroupExecutor.execute_parallel_group method
            mock_execute = mocker.patch("graflow.core.task.GroupExecutor.execute_parallel_group")

            group.run()

            # Verify backend_config was passed
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            assert call_args[1]["backend_config"] == {"thread_count": 4}
