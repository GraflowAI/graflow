"""Pytest-style tests for GroupExecutor functionality."""

from typing import List, cast

import pytest

from graflow.coordination.coordinator import CoordinationBackend, TaskCoordinator
from graflow.coordination.executor import GroupExecutor
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper
from graflow.queue.base import TaskSpec as QueueTaskSpec
from graflow.queue.base import TaskStatus


class TestTaskSpec:
    """Test cases for TaskSpec class."""

    def test_task_spec_creation(self):
        """Test TaskSpec creation and properties."""

        graph = TaskGraph()
        exec_context = ExecutionContext(graph)
        task = TaskWrapper("test_task", lambda: "test_result", register_to_context=False)
        spec = QueueTaskSpec(task, exec_context)

        assert spec.task_id == "test_task"
        assert spec.execution_context == exec_context
        assert spec.executable is task
        assert spec.status == TaskStatus.READY
        assert spec.group_id is None

    def test_task_retry_metadata(self):
        """Retry metadata is tracked."""
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)
        task = TaskWrapper("retry_task", lambda: None, register_to_context=False)
        spec = QueueTaskSpec(task, exec_context, max_retries=2)

        assert spec.can_retry()
        spec.increment_retry("boom")
        assert spec.retry_count == 1
        assert spec.last_error == "boom"


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

    def test_execute_parallel_group_direct(self, caplog: pytest.LogCaptureFixture):
        """Test direct execution without coordination."""
        import logging

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

        for t in tasks:
            t.set_execution_context(exec_context)
            graph.add_node(t)

        with caplog.at_level(logging.DEBUG):
            executor.execute_parallel_group("test_group", tasks, exec_context, backend="direct")

        # Verify tasks were executed
        assert "task1" in results
        assert "task2" in results

        # Verify log output
        assert "Running parallel group: test_group" in caplog.text
        assert "Direct tasks:" in caplog.text

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

        for t in tasks:
            t.set_execution_context(exec_context)
            graph.add_node(t)

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

        for t in tasks:
            t.set_execution_context(exec_context)
            graph.add_node(t)

        coordinator_mock = mocker.Mock(spec=TaskCoordinator)
        mocker.patch("graflow.coordination.executor.ThreadingCoordinator", return_value=coordinator_mock)

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

        for t in tasks:
            t.set_execution_context(exec_context)
            graph.add_node(t)

        queue_mock = mocker.Mock()
        coordinator_mock = mocker.Mock(spec=TaskCoordinator)

        mocker.patch("graflow.coordination.executor.DistributedTaskQueue", return_value=queue_mock)
        mocker.patch("graflow.coordination.executor.RedisCoordinator", return_value=coordinator_mock)

        executor.execute_parallel_group(
            "test_group",
            tasks,
            exec_context,
            backend="redis",
            backend_config={"host": "test-host", "port": 1234, "db": 2, "key_prefix": "prefix"},
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
        graph = TaskGraph()
        exec_context = ExecutionContext(graph)

        coordinator = GroupExecutor._create_coordinator(
            CoordinationBackend.THREADING, {"thread_count": 4}, exec_context
        )

        assert isinstance(coordinator, TaskCoordinator)

    def test_redis_import_error_handling(self, mocker):
        """Test Redis import error handling."""
        from graflow.core.decorators import task
        from graflow.core.task import Executable

        # Mock DistributedTaskQueue to simulate import error
        mocker.patch("graflow.coordination.executor.DistributedTaskQueue", side_effect=ImportError("redis missing"))

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
