"""Tests for ParallelGroup.with_execution() method."""

from graflow.coordination.coordinator import CoordinationBackend
from graflow.coordination.executor import GroupExecutor
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

            result = group.with_execution(
                backend=CoordinationBackend.REDIS
            ).set_group_name("my_group")

            assert isinstance(result, ParallelGroup)
            assert result.task_id == "my_group"
            assert result._execution_config["backend"] == CoordinationBackend.REDIS

    def test_with_execution_both_params(self):
        """Test with_execution() with both backend and backend_config."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            group.with_execution(
                backend=CoordinationBackend.THREADING,
                backend_config={"thread_count": 2}
            )

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

    def test_run_uses_configured_executor(self, mocker):
        """Test that run() uses executor from with_execution()."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            group.with_execution(backend=CoordinationBackend.THREADING)

            graph = TaskGraph()
            context = ExecutionContext(graph)
            group.set_execution_context(context)

            # Mock GroupExecutor to verify it's created correctly
            mock_executor = mocker.MagicMock(spec=GroupExecutor)
            mock_executor_class = mocker.patch(
                'graflow.core.task.GroupExecutor',
                return_value=mock_executor
            )

            group.run()

            # Verify GroupExecutor was created with correct backend
            mock_executor_class.assert_called_once_with(
                CoordinationBackend.THREADING, {}
            )
            mock_executor.execute_parallel_group.assert_called_once()
            _, kwargs = mock_executor.execute_parallel_group.call_args
            assert kwargs["policy"] == group._execution_config["policy"]

    def test_run_uses_context_executor_when_backend_none(self, mocker):
        """Test that run() uses context.group_executor when backend is None."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            # Don't call with_execution() - backend remains None
            assert group._execution_config["backend"] is None

            graph = TaskGraph()
            context = ExecutionContext(graph)
            context.group_executor = mocker.MagicMock(spec=GroupExecutor)
            group.set_execution_context(context)

            group.run()

            # Verify context.group_executor was used
            context.group_executor.execute_parallel_group.assert_called_once() # type: ignore
            _, kwargs = context.group_executor.execute_parallel_group.call_args # type: ignore
            assert kwargs["policy"] == group._execution_config["policy"] # type: ignore

    def test_run_with_execution_overrides_context_executor(self, mocker):
        """Test that with_execution() takes priority over context.group_executor."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            # Set backend via with_execution()
            group.with_execution(backend=CoordinationBackend.DIRECT)

            graph = TaskGraph()
            context = ExecutionContext(graph)
            # Set context.group_executor (should be ignored)
            context.group_executor = mocker.MagicMock(spec=GroupExecutor)
            group.set_execution_context(context)

            # Mock GroupExecutor to verify configured one is used
            mock_executor = mocker.MagicMock(spec=GroupExecutor)
            mock_executor_class = mocker.patch(
                'graflow.core.task.GroupExecutor',
                return_value=mock_executor
            )

            group.run()

            # Verify configured executor is used, not context.group_executor
            mock_executor_class.assert_called_once_with(
                CoordinationBackend.DIRECT, {}
            )
            mock_executor.execute_parallel_group.assert_called_once()
            _, kwargs = mock_executor.execute_parallel_group.call_args
            assert kwargs["policy"] == group._execution_config["policy"]
            context.group_executor.execute_parallel_group.assert_not_called() # type: ignore

    def test_run_uses_default_when_no_backend_and_no_context_executor(self, mocker):
        """Test that run() uses default executor when backend is None and no context executor."""
        with WorkflowContext("test"):
            task1 = Task("task1")
            task2 = Task("task2")
            group = ParallelGroup([task1, task2])

            # Don't call with_execution() - backend remains None
            assert group._execution_config["backend"] is None

            graph = TaskGraph()
            context = ExecutionContext(graph)
            # Don't set context.group_executor
            assert context.group_executor is None
            group.set_execution_context(context)

            # Mock GroupExecutor to verify default is created
            mock_executor = mocker.MagicMock(spec=GroupExecutor)
            mock_executor_class = mocker.patch(
                'graflow.core.task.GroupExecutor',
                return_value=mock_executor
            )

            group.run()

            # Verify default GroupExecutor() is created
            mock_executor_class.assert_called_once_with()
            mock_executor.execute_parallel_group.assert_called_once()
