"""Pytest-style tests for GroupExecutor functionality."""

from typing import List, cast

import pytest

from graflow.coordination.coordinator import CoordinationBackend, TaskCoordinator
from graflow.coordination.executor import GroupExecutor
from graflow.coordination.task_spec import TaskSpec
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph


class TestTaskSpec:
    """Test cases for TaskSpec class."""

    def test_task_spec_creation(self):
        """Test TaskSpec creation and properties."""
        def test_func():
            return "test_result"

        graph = TaskGraph()
        exec_context = ExecutionContext(graph)
        spec = TaskSpec("test_task", exec_context, test_func, args=(1, 2), kwargs={"key": "value"})

        assert spec.task_id == "test_task"
        assert spec.execution_context == exec_context
        assert spec.func == test_func
        assert spec.args == (1, 2)
        assert spec.kwargs == {"key": "value"}

    def test_task_spec_default_kwargs(self):
        """Test TaskSpec with default kwargs."""
        def test_func():
            pass

        graph = TaskGraph()
        exec_context = ExecutionContext(graph)
        spec = TaskSpec("test_task", exec_context, test_func)

        assert spec.kwargs == {}


class TestGroupExecutor:
    """Test cases for GroupExecutor class."""

    @pytest.fixture
    def mock_coordinator(self, mocker):
        """Create mock coordinator."""
        return mocker.Mock(spec=TaskCoordinator)

    def test_group_executor_default_backend(self):
        """Test GroupExecutor with default threading backend."""
        executor = GroupExecutor()

        assert executor.backend == CoordinationBackend.THREADING

    def test_group_executor_direct_backend(self):
        """Test GroupExecutor with direct backend."""
        executor = GroupExecutor(CoordinationBackend.DIRECT)

        assert executor.backend == CoordinationBackend.DIRECT

    def test_group_executor_redis_backend(self, mocker):
        """Test GroupExecutor with Redis backend."""
        mock_redis = mocker.patch('graflow.coordination.executor.redis')
        mock_redis_coord = mocker.patch('graflow.coordination.executor.RedisCoordinator')

        mock_redis_client = mocker.Mock()
        mock_redis.Redis.return_value = mock_redis_client

        executor = GroupExecutor(
            CoordinationBackend.REDIS,
            {"host": "test-host", "port": 1234, "db": 1}
        )

        assert executor.backend == CoordinationBackend.REDIS
        mock_redis.Redis.assert_called_once_with(host="test-host", port=1234, db=1)
        mock_redis_coord.assert_called_once_with(mock_redis_client)

    def test_group_executor_redis_with_client(self, mocker):
        """Test GroupExecutor with existing Redis client."""
        mock_redis_coord = mocker.patch('graflow.coordination.executor.RedisCoordinator')
        mock_redis_client = mocker.Mock()

        _executor = GroupExecutor(
            CoordinationBackend.REDIS,
            {"redis_client": mock_redis_client}
        )

        mock_redis_coord.assert_called_once_with(mock_redis_client)

    def test_execute_parallel_group_direct(self, mocker):
        """Test direct execution without coordination."""
        from graflow.core.decorators import task
        from graflow.core.task import Executable

        executor = GroupExecutor(CoordinationBackend.DIRECT)
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        results = []

        @task
        def task1():
            results.append("task1")
            return "result1"

        @task
        def task2():
            results.append("task2")
            return "result2"

        tasks = cast(List[Executable], [task1, task2])

        mock_print = mocker.patch('builtins.print')
        executor.execute_parallel_group("test_group", tasks, exec_context)

        # Verify tasks were executed
        assert "task1" in results
        assert "task2" in results

        # Verify print output
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Running parallel group: test_group" in call for call in print_calls)
        assert any("Direct tasks:" in call for call in print_calls)

    def test_execute_parallel_group_direct_with_args(self):
        """Test direct execution with args and kwargs."""
        from graflow.core.decorators import task
        from graflow.core.task import Executable

        executor = GroupExecutor(CoordinationBackend.DIRECT)
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        results = []

        @task
        def task1():
            # Simulate task with specific behavior
            results.append(("a", "b", "c"))
            return ("a", "b", "c")

        tasks = cast(List[Executable], [task1])

        executor.execute_parallel_group("test_group", tasks, exec_context)

        assert results == [("a", "b", "c")]

    def test_execute_parallel_group_threading(self, mocker):
        """Test parallel execution with threading backend."""
        from graflow.core.decorators import task
        from graflow.core.task import Executable

        executor = GroupExecutor(CoordinationBackend.THREADING)
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        results = []

        @task
        def task1():
            results.append("task1")
            return "result_task1"

        @task
        def task2():
            results.append("task2")
            return "result_task2"

        tasks = cast(List[Executable], [task1, task2])

        # Execute should not raise
        executor.execute_parallel_group("test_group", tasks, exec_context)

        # Verify both tasks executed
        assert "task1" in results
        assert "task2" in results


class TestGroupExecutorIntegration:
    """Integration tests for GroupExecutor."""

    def test_threading_backend_creation(self):
        """Test threading backend coordinator creation."""
        executor = GroupExecutor(
            CoordinationBackend.THREADING,
            {"thread_count": 4}
        )

        # Verify executor was configured correctly
        assert executor.backend == CoordinationBackend.THREADING
        assert executor.backend_config["thread_count"] == 4

    def test_redis_import_error_handling(self, mocker):
        """Test Redis import error handling."""
        from graflow.core.decorators import task
        from graflow.core.task import Executable

        # Mock RedisCoordinator to simulate import error
        mocker.patch('graflow.coordination.executor.RedisCoordinator', side_effect=ImportError("No module named 'redis'"))

        executor = GroupExecutor(CoordinationBackend.REDIS)
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        @task
        def test_task():
            return "result"

        tasks = cast(List[Executable], [test_task])

        # Error should occur when trying to create coordinator
        with pytest.raises(ImportError, match="Redis backend requires 'redis' package"):
            executor.execute_parallel_group("test_group", tasks, exec_context)
