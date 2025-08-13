"""Integration tests for Redis coordinator with real Redis server."""

import json
import threading
import time

import pytest

from graflow.coordination.redis import RedisCoordinator
from graflow.coordination.task_spec import TaskSpec


@pytest.mark.integration
class TestRedisCoordinatorIntegration:
    """Integration tests for RedisCoordinator with real Redis server."""

    def test_barrier_synchronization(self, clean_redis):
        """Test barrier synchronization with real Redis."""
        coordinator = RedisCoordinator(clean_redis)

        # Create barrier for 2 participants
        barrier_id = "test_barrier"
        coordinator.create_barrier(barrier_id, 2)

        # Simulate first participant waiting (in separate thread)
        results = []

        def first_participant():
            """First participant waits at barrier."""
            result = coordinator.wait_barrier(barrier_id, timeout=5)
            results.append(("first", result))

        thread = threading.Thread(target=first_participant)
        thread.start()

        # Small delay to ensure first participant is waiting
        time.sleep(0.1)

        # Second participant completes the barrier
        result = coordinator.wait_barrier("test_barrier", timeout=5)
        thread.join()

        # Both participants should succeed
        assert result is True
        assert len(results) == 1
        assert results[0] == ("first", True)

        # Clean up
        coordinator.cleanup_barrier("test_barrier")

    def test_barrier_timeout(self, clean_redis):
        """Test barrier timeout with real Redis."""
        coordinator = RedisCoordinator(clean_redis)

        # Create barrier for 2 participants but only one waits
        coordinator.create_barrier("timeout_barrier", 2)

        start_time = time.time()
        result = coordinator.wait_barrier("timeout_barrier", timeout=1)
        elapsed_time = time.time() - start_time

        assert result is False
        assert elapsed_time >= 0.9  # Should take at least close to timeout

        coordinator.cleanup_barrier("timeout_barrier")

    def test_task_dispatch_and_queue(self, clean_redis):
        """Test task dispatching to Redis queue."""
        coordinator = RedisCoordinator(clean_redis)

        def test_function():
            return "test_result"

        task_spec = TaskSpec("test_task", test_function, args=(1, 2), kwargs={"key": "value"})

        # Initially queue should be empty
        assert coordinator.get_queue_size("test_group") == 0

        # Dispatch task
        coordinator.dispatch_task(task_spec, "test_group")

        # Queue should have one task
        assert coordinator.get_queue_size("test_group") == 1

        # Retrieve task data from queue
        queue_key = "task_queue:test_group"
        task_data_json = clean_redis.rpop(queue_key)
        task_data = json.loads(task_data_json)

        assert task_data["task_id"] == "test_task"
        assert task_data["group_id"] == "test_group"
        assert task_data["args"] == [1, 2]  # JSON converts tuples to lists
        assert task_data["kwargs"] == {"key": "value"}
        assert "timestamp" in task_data

    def test_multiple_task_dispatch(self, clean_redis):
        """Test dispatching multiple tasks to queue."""
        coordinator = RedisCoordinator(clean_redis)

        def task_func(n):
            return f"result_{n}"

        # Dispatch multiple tasks
        for i in range(5):
            task_spec = TaskSpec(f"task_{i}", task_func, args=(i,))
            coordinator.dispatch_task(task_spec, "multi_group")

        # Check queue size
        assert coordinator.get_queue_size("multi_group") == 5

        # Tasks should be retrievable in FIFO order (lpush + rpop)
        queue_key = "task_queue:multi_group"
        for i in range(5):
            task_data_json = clean_redis.rpop(queue_key)
            task_data = json.loads(task_data_json)
            assert task_data["task_id"] == f"task_{i}"

    def test_queue_operations(self, clean_redis):
        """Test queue size and clear operations."""
        coordinator = RedisCoordinator(clean_redis)

        def test_func():
            pass

        # Add some tasks
        for i in range(3):
            task_spec = TaskSpec(f"task_{i}", test_func)
            coordinator.dispatch_task(task_spec, "queue_test")

        assert coordinator.get_queue_size("queue_test") == 3

        # Clear queue
        coordinator.clear_queue("queue_test")
        assert coordinator.get_queue_size("queue_test") == 0

    def test_barrier_cleanup(self, clean_redis):
        """Test proper barrier cleanup."""
        coordinator = RedisCoordinator(clean_redis)

        # Create barrier
        _barrier_id = coordinator.create_barrier("cleanup_test", 1)

        # Verify barrier exists in Redis
        barrier_key = "barrier:cleanup_test"
        expected_key = f"{barrier_key}:expected"

        assert clean_redis.exists(barrier_key) or clean_redis.exists(expected_key)
        assert "cleanup_test" in coordinator.active_barriers

        # Cleanup barrier
        coordinator.cleanup_barrier("cleanup_test")

        # Verify barrier is cleaned up
        assert not clean_redis.exists(barrier_key)
        assert not clean_redis.exists(expected_key)
        assert "cleanup_test" not in coordinator.active_barriers

    def test_concurrent_barrier_operations(self, clean_redis):
        """Test concurrent barrier operations."""

        coordinator = RedisCoordinator(clean_redis)

        # Create barrier for 3 participants
        coordinator.create_barrier("concurrent_test", 3)

        results = []

        def participant(participant_id):
            result = coordinator.wait_barrier("concurrent_test", timeout=5)
            results.append((participant_id, result, time.time()))

        # Start 3 participants
        threads = []
        for i in range(3):
            thread = threading.Thread(target=participant, args=(i,))
            thread.start()
            threads.append(thread)

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All participants should succeed
        assert len(results) == 3
        for _participant_id, result, _timestamp in results:
            assert result is True

        # All should complete around the same time (within 1 second)
        timestamps = [timestamp for _, _, timestamp in results]
        time_spread = max(timestamps) - min(timestamps)
        assert time_spread < 1.0

        coordinator.cleanup_barrier("concurrent_test")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
