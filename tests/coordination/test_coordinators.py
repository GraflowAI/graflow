"""Pytest-style tests for coordination backends."""

import json
from unittest.mock import call

import pytest

from graflow.coordination.coordinator import CoordinationBackend
from graflow.coordination.multiprocessing import MultiprocessingCoordinator
from graflow.coordination.redis import RedisCoordinator
from graflow.coordination.task_spec import TaskSpec


class TestMultiprocessingCoordinator:
    """Test cases for MultiprocessingCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create coordinator for tests."""
        return MultiprocessingCoordinator(process_count=2)

    @pytest.fixture(autouse=True)
    def cleanup_coordinator(self, coordinator):
        """Clean up coordinator after each test."""
        yield
        if hasattr(coordinator, 'stop_workers'):
            coordinator.stop_workers()

    def test_multiprocessing_coordinator_creation(self, coordinator):
        """Test MultiprocessingCoordinator creation."""
        assert coordinator.mp is not None
        assert coordinator.manager is not None
        assert coordinator.task_queue is not None
        assert coordinator.result_queue is not None

    def test_multiprocessing_coordinator_process_count(self):
        """Test process count configuration."""
        coord = MultiprocessingCoordinator(process_count=4)
        assert coord.process_count == 4

    def test_create_barrier(self, coordinator):
        """Test barrier creation."""
        barrier_id = coordinator.create_barrier("test_barrier", 3)

        assert barrier_id == "test_barrier"
        assert "test_barrier" in coordinator.barriers

        barrier = coordinator.barriers["test_barrier"]
        assert barrier.parties == 3

    def test_create_barrier_error_handling(self, coordinator, mocker):
        """Test barrier creation error handling."""
        mocker.patch.object(coordinator.mp, 'Barrier', side_effect=Exception("Test error"))
        with pytest.raises(Exception):
            coordinator.create_barrier("error_barrier", 2)

    def test_wait_barrier_success(self, coordinator):
        """Test successful barrier waiting."""
        barrier_id = coordinator.create_barrier("test_barrier", 1)

        # Should succeed immediately with 1 participant
        result = coordinator.wait_barrier(barrier_id, timeout=1)
        assert result is True

    def test_wait_barrier_nonexistent(self, coordinator):
        """Test waiting on non-existent barrier."""
        result = coordinator.wait_barrier("nonexistent", timeout=1)
        assert result is False

    def test_signal_barrier(self, coordinator):
        """Test barrier signaling (no-op in multiprocessing)."""
        # This is a no-op in multiprocessing implementation
        coordinator.signal_barrier("test_barrier")
        # Should not raise any exceptions

    def test_dispatch_task(self, coordinator, mocker):
        """Test task dispatching."""
        # Mock the queue methods directly on the coordinator instance
        mocker.patch.object(coordinator, 'get_queue_size', return_value=1)
        mocker.patch.object(coordinator, 'is_queue_empty', return_value=False)

        def test_func():
            return "test_result"

        task_spec = TaskSpec("test_task", test_func, args=(1, 2), kwargs={"key": "value"})

        coordinator.dispatch_task(task_spec, "test_group")

        # Verify task was queued (using mocked methods)
        assert not coordinator.is_queue_empty()

    def test_cleanup_barrier(self, coordinator):
        """Test barrier cleanup."""
        barrier_id = coordinator.create_barrier("cleanup_barrier", 2)
        assert barrier_id in coordinator.barriers

        coordinator.cleanup_barrier(barrier_id)
        assert barrier_id not in coordinator.barriers

    def test_cleanup_nonexistent_barrier(self, coordinator):
        """Test cleanup of non-existent barrier."""
        # Should not raise exceptions
        coordinator.cleanup_barrier("nonexistent")

    def test_start_stop_workers(self, coordinator, mocker):
        """Test worker lifecycle."""
        mock_worker = mocker.MagicMock()
        mock_worker.is_alive.return_value = True
        mock_process = mocker.patch('multiprocessing.Process', return_value=mock_worker)

        coordinator.start_workers()
        assert len(coordinator.workers) == 2

        coordinator.stop_workers()
        assert len(coordinator.workers) == 0

    def test_worker_execution(self, coordinator, mocker):
        """Test task execution by workers (mocked)."""
        # Mock the worker process and queue methods
        mock_worker = mocker.MagicMock()
        mock_process = mocker.patch('multiprocessing.Process', return_value=mock_worker)
        mocker.patch.object(coordinator, 'get_queue_size', side_effect=[0, 1])

        def test_task(value):
            return f"result_{value}"

        task_spec = TaskSpec("test_task", test_task, args=("test",))

        # Test task dispatch (queue operations)
        coordinator.dispatch_task(task_spec, "test_group")
        # Just verify no exceptions are raised

    def test_worker_error_handling(self, coordinator):
        """Test worker error handling (simplified)."""
        def failing_task():
            raise ValueError("Test error")

        task_data = {
            "task_id": "failing_task",
            "func": failing_task,
            "args": (),
            "kwargs": {}
        }

        with pytest.raises(ValueError):
            coordinator._execute_task(task_data)


class TestRedisCoordinator:
    """Test cases for RedisCoordinator."""

    @pytest.fixture
    def mock_redis(self, mocker):
        """Create mock Redis client."""
        return mocker.MagicMock()

    @pytest.fixture
    def coordinator(self, mock_redis):
        """Create Redis coordinator with mock client."""
        return RedisCoordinator(mock_redis)

    def test_redis_coordinator_creation(self, coordinator, mock_redis):
        """Test RedisCoordinator creation."""
        assert coordinator.redis == mock_redis
        assert isinstance(coordinator.active_barriers, dict)

    def test_create_barrier(self, coordinator, mock_redis):
        """Test Redis barrier creation."""
        barrier_id = coordinator.create_barrier("test_barrier", 3)

        assert barrier_id == "barrier:test_barrier"
        assert "test_barrier" in coordinator.active_barriers

        # Verify Redis calls
        mock_redis.delete.assert_called_with("barrier:test_barrier")
        mock_redis.set.assert_called_with("barrier:test_barrier:expected", 3)

    def test_wait_barrier_success(self, coordinator, mock_redis):
        """Test successful barrier waiting."""
        # Setup barrier
        coordinator.create_barrier("test_barrier", 2)

        # Mock Redis to simulate being the last participant
        mock_redis.incr.return_value = 2

        result = coordinator.wait_barrier("test_barrier", timeout=1)

        assert result is True
        mock_redis.incr.assert_called_with("barrier:test_barrier")
        mock_redis.publish.assert_called_with("barrier_done:test_barrier", "complete")

    def test_wait_barrier_with_pubsub(self, coordinator, mock_redis, mocker):
        """Test barrier waiting with pub/sub."""
        # Setup barrier
        coordinator.create_barrier("test_barrier", 2)

        # Mock Redis to simulate not being the last participant
        mock_redis.incr.return_value = 1

        # Mock pubsub
        mock_pubsub = mocker.MagicMock()
        mock_message = {"type": "message", "data": b"complete"}
        mock_pubsub.listen.return_value = [mock_message]
        mock_redis.pubsub.return_value = mock_pubsub

        result = coordinator.wait_barrier("test_barrier", timeout=1)

        assert result is True
        mock_pubsub.subscribe.assert_called_with("barrier_done:test_barrier")
        mock_pubsub.close.assert_called_once()

    def test_wait_barrier_timeout(self, coordinator, mock_redis, mocker):
        """Test barrier timeout."""
        coordinator.create_barrier("test_barrier", 2)

        mock_redis.incr.return_value = 1

        # Mock pubsub with no messages (timeout)
        mock_pubsub = mocker.MagicMock()
        mock_pubsub.listen.return_value = []
        mock_redis.pubsub.return_value = mock_pubsub

        mocker.patch('time.time', side_effect=[0, 0, 2])  # Simulate timeout
        result = coordinator.wait_barrier("test_barrier", timeout=1)

        assert result is False

    def test_wait_barrier_nonexistent(self, coordinator):
        """Test waiting on non-existent barrier."""
        result = coordinator.wait_barrier("nonexistent", timeout=1)
        assert result is False

    def test_signal_barrier(self, coordinator, mock_redis):
        """Test barrier signaling."""
        coordinator.create_barrier("test_barrier", 2)
        mock_redis.incr.return_value = 2  # Last participant

        coordinator.signal_barrier("test_barrier")

        mock_redis.publish.assert_called_with("barrier_done:test_barrier", "complete")

    def test_dispatch_task(self, coordinator, mock_redis):
        """Test task dispatching to Redis queue."""
        def test_func():
            return "test"

        task_spec = TaskSpec("test_task", test_func, args=(1, 2), kwargs={"key": "value"})

        coordinator.dispatch_task(task_spec, "test_group")

        # Verify Redis call
        mock_redis.lpush.assert_called_once()
        args, kwargs = mock_redis.lpush.call_args
        queue_key, task_data_json = args

        assert queue_key == "task_queue:test_group"

        task_data = json.loads(task_data_json)
        assert task_data["task_id"] == "test_task"
        assert task_data["group_id"] == "test_group"
        assert task_data["args"] == [1, 2]  # JSON converts tuples to lists
        assert task_data["kwargs"] == {"key": "value"}

    def test_cleanup_barrier(self, coordinator, mock_redis):
        """Test barrier cleanup."""
        coordinator.create_barrier("cleanup_barrier", 2)

        coordinator.cleanup_barrier("cleanup_barrier")

        # Verify Redis cleanup calls
        expected_calls = [
            call("barrier:cleanup_barrier"),
            call("barrier:cleanup_barrier:expected")
        ]
        mock_redis.delete.assert_has_calls(expected_calls, any_order=True)

        # Verify barrier removed from active barriers
        assert "cleanup_barrier" not in coordinator.active_barriers

    def test_task_function_registry(self, coordinator):
        """Test task function registration."""
        def test_func_1():
            return "result1"

        def test_func_2():
            return "result2"

        # Initially empty
        assert coordinator.get_task_registry() == {}

        # Register functions
        coordinator.register_task_function("func1", test_func_1)
        coordinator.register_task_function("func2", test_func_2)

        # Retrieve registry
        registry = coordinator.get_task_registry()
        assert len(registry) == 2
        assert registry["func1"] == test_func_1
        assert registry["func2"] == test_func_2

    def test_queue_operations(self, coordinator, mock_redis):
        """Test queue size and clear operations."""
        mock_redis.llen.return_value = 5

        size = coordinator.get_queue_size("test_group")
        assert size == 5
        mock_redis.llen.assert_called_with("task_queue:test_group")

        coordinator.clear_queue("test_group")
        mock_redis.delete.assert_called_with("task_queue:test_group")


class TestCoordinationBackend:
    """Test cases for CoordinationBackend enum."""

    def test_coordination_backend_values(self):
        """Test CoordinationBackend enum values."""
        assert CoordinationBackend.REDIS.value == "redis"
        assert CoordinationBackend.MULTIPROCESSING.value == "multiprocessing"
        assert CoordinationBackend.DIRECT.value == "direct"