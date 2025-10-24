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

    def test_resolve_backend_defaults_to_threading(self):
        """Ensure the default backend resolves to THREADING."""
        executor = GroupExecutor()
        assert executor._resolve_backend(None) == CoordinationBackend.THREADING

    def test_resolve_backend_accepts_string(self):
        """Ensure string backend names resolve to CoordinationBackend members."""
        executor = GroupExecutor()
        assert executor._resolve_backend("redis") == CoordinationBackend.REDIS
        assert executor._resolve_backend("DIRECT") == CoordinationBackend.DIRECT

    def test_execute_parallel_group_direct(self, mocker):
        """Test direct execution without coordination."""
        from graflow.core.decorators import task
        from graflow.core.task import Executable

        executor = GroupExecutor()
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
        executor.execute_parallel_group("test_group", tasks, exec_context, backend="direct")

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

        executor = GroupExecutor()
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        results = []

        @task
        def task1():
            # Simulate task with specific behavior
            results.append(("a", "b", "c"))
            return ("a", "b", "c")

        tasks = cast(List[Executable], [task1])

        executor.execute_parallel_group("test_group", tasks, exec_context, backend=CoordinationBackend.DIRECT)

        assert results == [("a", "b", "c")]

    def test_execute_parallel_group_threading(self, mocker):
        """Test parallel execution with threading backend."""
        from graflow.core.decorators import task
        from graflow.core.task import Executable

        executor = GroupExecutor()
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        @task
        def task1():
            return "result_task1"

        @task
        def task2():
            return "result_task2"

        tasks = cast(List[Executable], [task1, task2])

        coordinator_mock = mocker.Mock(spec=TaskCoordinator)
        mocker.patch('graflow.coordination.executor.ThreadingCoordinator', return_value=coordinator_mock)

        # Execute should not raise and should delegate to coordinator
        executor.execute_parallel_group("test_group", tasks, exec_context, backend="threading")

        coordinator_mock.execute_group.assert_called_once()

    def test_execute_parallel_group_redis(self, mocker):
        """Ensure Redis backend initializes the queue and coordinator."""
        from graflow.core.decorators import task
        from graflow.core.task import Executable

        executor = GroupExecutor()
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        @task
        def task1():
            return "ok"

        tasks = cast(List[Executable], [task1])

        queue_mock = mocker.Mock()
        coordinator_mock = mocker.Mock(spec=TaskCoordinator)

        mocker.patch('graflow.coordination.executor.RedisTaskQueue', return_value=queue_mock)
        mocker.patch('graflow.coordination.executor.RedisCoordinator', return_value=coordinator_mock)

        executor.execute_parallel_group(
            "test_group",
            tasks,
            exec_context,
            backend="redis",
            backend_config={"host": "test-host", "port": 1234, "db": 2, "key_prefix": "prefix"}
        )

        coordinator_mock.execute_group.assert_called_once_with("test_group", tasks, exec_context, mocker.ANY)

    def test_execute_parallel_group_invalid_backend(self):
        executor = GroupExecutor()
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        with pytest.raises(ValueError):
            executor.execute_parallel_group("group", [], exec_context, backend="unsupported")


class TestGroupExecutorIntegration:
    """Integration tests for GroupExecutor."""

    def test_create_threading_coordinator(self):
        executor = GroupExecutor()
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        coordinator = executor._create_coordinator(
            CoordinationBackend.THREADING,
            {"thread_count": 4},
            exec_context
        )

        assert isinstance(coordinator, TaskCoordinator)

    def test_redis_import_error_handling(self, mocker):
        """Test Redis import error handling."""
        from graflow.core.decorators import task
        from graflow.core.task import Executable

        # Mock RedisCoordinator to simulate import error
        mocker.patch('graflow.coordination.executor.RedisTaskQueue', side_effect=ImportError("redis missing"))

        executor = GroupExecutor()
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        @task
        def test_task():
            return "result"

        tasks = cast(List[Executable], [test_task])

        # Error should occur when trying to create coordinator
        with pytest.raises(ImportError, match="Redis backend requires 'redis' package"):
            executor.execute_parallel_group("test_group", tasks, exec_context, backend="redis")
