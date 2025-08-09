"""Pytest-style tests for GroupExecutor functionality."""

from typing import Any

import pytest

from graflow.coordination.coordinator import CoordinationBackend, TaskCoordinator
from graflow.coordination.executor import GroupExecutor
from graflow.coordination.task_spec import TaskSpec


class TestTaskSpec:
    """Test cases for TaskSpec class."""

    def test_task_spec_creation(self):
        """Test TaskSpec creation and properties."""
        def test_func():
            return "test_result"

        spec = TaskSpec("test_task", test_func, args=(1, 2), kwargs={"key": "value"})

        assert spec.task_id == "test_task"
        assert spec.func == test_func
        assert spec.args == (1, 2)
        assert spec.kwargs == {"key": "value"}

    def test_task_spec_default_kwargs(self):
        """Test TaskSpec with default kwargs."""
        def test_func():
            pass

        spec = TaskSpec("test_task", test_func)

        assert spec.kwargs == {}


class TestGroupExecutor:
    """Test cases for GroupExecutor class."""

    @pytest.fixture
    def mock_coordinator(self, mocker):
        """Create mock coordinator."""
        return mocker.Mock(spec=TaskCoordinator)

    def test_group_executor_default_backend(self, mocker):
        """Test GroupExecutor with default multiprocessing backend."""
        mock_mp = mocker.patch('graflow.coordination.executor.MultiprocessingCoordinator')
        executor = GroupExecutor()

        assert executor.backend == CoordinationBackend.MULTIPROCESSING
        mock_mp.assert_called_once_with(None)

    def test_group_executor_direct_backend(self):
        """Test GroupExecutor with direct backend."""
        executor = GroupExecutor(CoordinationBackend.DIRECT)

        assert executor.backend == CoordinationBackend.DIRECT
        assert executor.coordinator is None

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

    def test_set_execution_context(self, mocker):
        """Test setting execution context."""
        executor = GroupExecutor(CoordinationBackend.DIRECT)
        mock_context = mocker.Mock()

        executor.set_execution_context(mock_context)

        assert executor.execution_context == mock_context

    def test_execute_parallel_group_direct(self, mocker):
        """Test direct execution without coordination."""
        executor = GroupExecutor(CoordinationBackend.DIRECT)

        results = []
        def task1_func():
            results.append("task1")

        def task2_func():
            results.append("task2")

        tasks = [
            TaskSpec("task1", task1_func),
            TaskSpec("task2", task2_func)
        ]

        mock_print = mocker.patch('builtins.print')
        executor.execute_parallel_group("test_group", tasks)

        # Verify tasks were executed
        assert results == ["task1", "task2"]

        # Verify print output
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Running parallel group: test_group" in call for call in print_calls)
        assert any("Direct tasks:" in call for call in print_calls)

    def test_execute_parallel_group_direct_with_args(self):
        """Test direct execution with args and kwargs."""
        executor = GroupExecutor(CoordinationBackend.DIRECT)

        results = []
        def task_func(arg1, arg2, kwarg1=None):
            results.append((arg1, arg2, kwarg1))

        tasks = [
            TaskSpec("task1", task_func, args=("a", "b"), kwargs={"kwarg1": "c"})
        ]

        executor.execute_parallel_group("test_group", tasks)

        assert results == [("a", "b", "c")]

    def test_execute_parallel_group_with_coordinator(self, mock_coordinator: Any):
        """Test parallel execution with coordinator."""
        executor = GroupExecutor(CoordinationBackend.MULTIPROCESSING)
        executor.coordinator = mock_coordinator

        # Mock coordinator behavior
        mock_coordinator.create_barrier.return_value = "barrier_123"
        mock_coordinator.wait_barrier.return_value = True

        tasks = [
            TaskSpec("task1", lambda: None),
            TaskSpec("task2", lambda: None)
        ]

        executor.execute_parallel_group("test_group", tasks)

        # Verify coordinator calls
        mock_coordinator.create_barrier.assert_called_once_with("test_group", 2)
        assert mock_coordinator.dispatch_task.call_count == 2
        mock_coordinator.wait_barrier.assert_called_once_with("barrier_123")
        mock_coordinator.cleanup_barrier.assert_called_once_with("barrier_123")

    def test_execute_parallel_group_barrier_timeout(self, mock_coordinator: Any):
        """Test barrier timeout handling."""
        executor = GroupExecutor(CoordinationBackend.MULTIPROCESSING)
        executor.coordinator = mock_coordinator

        # Mock barrier timeout
        mock_coordinator.create_barrier.return_value = "barrier_123"
        mock_coordinator.wait_barrier.return_value = False

        tasks = [TaskSpec("task1", lambda: None)]

        with pytest.raises(TimeoutError):
            executor.execute_parallel_group("test_group", tasks)

        # Verify cleanup still happens
        mock_coordinator.cleanup_barrier.assert_called_once_with("barrier_123")

    def test_execute_parallel_group_coordinator_exception(self, mock_coordinator: Any):
        """Test exception handling during coordinator execution."""
        executor = GroupExecutor(CoordinationBackend.MULTIPROCESSING)
        executor.coordinator = mock_coordinator

        # Mock coordinator exception
        mock_coordinator.create_barrier.return_value = "barrier_123"
        mock_coordinator.dispatch_task.side_effect = RuntimeError("Test exception")

        tasks = [TaskSpec("task1", lambda: None)]

        with pytest.raises(RuntimeError):
            executor.execute_parallel_group("test_group", tasks)

        # Verify cleanup still happens
        mock_coordinator.cleanup_barrier.assert_called_once_with("barrier_123")

    def test_execute_parallel_group_no_coordinator_for_parallel(self):
        """Test assertion when coordinator is None for parallel execution."""
        executor = GroupExecutor(CoordinationBackend.MULTIPROCESSING)
        executor.coordinator = None  # Force None

        tasks = [TaskSpec("task1", lambda: None)]

        with pytest.raises(AssertionError):
            executor.execute_parallel_group("test_group", tasks)


class TestGroupExecutorIntegration:
    """Integration tests for GroupExecutor."""

    def test_multiprocessing_backend_creation(self, mocker):
        """Test multiprocessing backend coordinator creation."""
        mock_mp = mocker.patch('graflow.coordination.executor.MultiprocessingCoordinator')
        _executor = GroupExecutor(
            CoordinationBackend.MULTIPROCESSING,
            {"process_count": 4}
        )

        mock_mp.assert_called_once_with(4)

    def test_redis_import_error_handling(self, mocker):
        """Test Redis import error handling."""
        # Mock both redis import and RedisCoordinator to simulate import error
        mocker.patch('graflow.coordination.executor.RedisCoordinator', side_effect=ImportError("No module named 'redis'"))
        with pytest.raises(ImportError, match="Redis backend requires 'redis' package"):
            GroupExecutor(CoordinationBackend.REDIS)
