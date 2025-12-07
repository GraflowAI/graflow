"""Integration tests for Redis coordinator with real Redis server."""

import json
import threading
import time

import pytest

from graflow.coordination.redis_coordinator import RedisCoordinator
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper
from graflow.queue.distributed import DistributedTaskQueue


def create_coordinator(redis_client, key_prefix: str = "test") -> tuple[RedisCoordinator, ExecutionContext, DistributedTaskQueue]:
    """Helper to create coordinator with backing queue/context."""
    graph = TaskGraph()
    context = ExecutionContext(graph)
    queue = DistributedTaskQueue(redis_client=redis_client, key_prefix=key_prefix)
    return RedisCoordinator(queue), context, queue


@pytest.mark.integration
class TestRedisCoordinatorIntegration:
    """Integration tests for RedisCoordinator with real Redis server."""

    def test_barrier_synchronization(self, clean_redis):
        """Test barrier synchronization with real Redis.

        BSP model: Producer waits, workers increment via record_task_completion.
        """
        from graflow.coordination.redis_coordinator import record_task_completion

        coordinator, _, queue = create_coordinator(clean_redis)

        # Create barrier for 2 tasks
        barrier_id = "test_barrier"
        coordinator.create_barrier(barrier_id, 2)

        # Producer waits in separate thread
        results = []

        def producer_wait():
            """Producer waits for workers to complete."""
            result = coordinator.wait_barrier(barrier_id, timeout=5)
            results.append(("producer", result))

        producer_thread = threading.Thread(target=producer_wait)
        producer_thread.start()

        # Small delay to ensure producer is waiting
        time.sleep(0.1)

        # Simulate workers completing tasks
        record_task_completion(
            clean_redis, queue.key_prefix, "task_1", barrier_id, True
        )
        record_task_completion(
            clean_redis, queue.key_prefix, "task_2", barrier_id, True
        )

        producer_thread.join()

        # Producer should succeed
        assert len(results) == 1
        assert results[0] == ("producer", True)

        # Clean up
        coordinator.cleanup_barrier("test_barrier")

    def test_barrier_timeout(self, clean_redis):
        """Test barrier timeout with real Redis.

        Producer waits but workers never complete.
        """
        coordinator, _, _ = create_coordinator(clean_redis)

        # Create barrier for 2 tasks but no workers complete
        coordinator.create_barrier("timeout_barrier", 2)

        start_time = time.time()
        result = coordinator.wait_barrier("timeout_barrier", timeout=1)
        elapsed_time = time.time() - start_time

        assert result is False
        assert elapsed_time >= 0.9  # Should take at least close to timeout

        coordinator.cleanup_barrier("timeout_barrier")

    def test_task_dispatch_and_queue(self, clean_redis):
        """Test task dispatching to Redis queue."""
        coordinator, context, queue = create_coordinator(clean_redis, key_prefix="integration")
        graph = context.graph
        task = TaskWrapper("test_task", lambda: "test_result", register_to_context=False)
        graph.add_node(task)
        # Producer would set graph_hash via GraphStore.save()
        context.graph_hash = coordinator.graph_store.save(graph)
        task.set_execution_context(context)

        assert coordinator.get_queue_size("test_group") == 0

        coordinator.dispatch_task(task, "test_group")

        # Queue should have one task
        assert coordinator.get_queue_size("test_group") == 1

        # Retrieve task data from queue
        task_data_json = clean_redis.rpop(queue.queue_key)
        task_data = json.loads(task_data_json)

        assert task_data["task_id"] == "test_task"
        assert task_data["group_id"] == "test_group"
        assert "created_at" in task_data
        assert task_data["graph_hash"] == context.graph_hash

    def test_multiple_task_dispatch(self, clean_redis):
        """Test dispatching multiple tasks to queue."""
        coordinator, context, queue = create_coordinator(clean_redis, key_prefix="integration-multi")
        for i in range(5):
            task = TaskWrapper(f"task_{i}", lambda n=i: f"result_{n}", register_to_context=False)
            context.graph.add_node(task)
        context.graph_hash = coordinator.graph_store.save(context.graph)
        for node in context.graph.nodes:
            context.graph.get_node(node).set_execution_context(context)
        for task_id in context.graph.nodes:
            task = context.graph.get_node(task_id)
            coordinator.dispatch_task(task, "multi_group")

        # Check queue size
        assert coordinator.get_queue_size("multi_group") == 5

        # Tasks should be retrievable in FIFO order (lpush + rpop)
        for i in range(5):
            task_data_json = clean_redis.rpop(queue.queue_key)
            task_data = json.loads(task_data_json)
            assert task_data["task_id"] == f"task_{i}"

    def test_queue_operations(self, clean_redis):
        """Test queue size and clear operations."""
        coordinator, context, _ = create_coordinator(clean_redis, key_prefix="integration-queue")
        for i in range(3):
            task = TaskWrapper(f"task_{i}", lambda n=i: n, register_to_context=False)
            context.graph.add_node(task)
        context.graph_hash = coordinator.graph_store.save(context.graph)
        for node in context.graph.nodes:
            task = context.graph.get_node(node)
            task.set_execution_context(context)
            coordinator.dispatch_task(task, "queue_test")

        assert coordinator.get_queue_size("queue_test") == 3

        # Clear queue
        coordinator.clear_queue("queue_test")
        assert coordinator.get_queue_size("queue_test") == 0

    def test_barrier_cleanup(self, clean_redis):
        """Test proper barrier cleanup."""
        coordinator, _, queue = create_coordinator(clean_redis, key_prefix="integration-cleanup")

        _barrier_id = coordinator.create_barrier("cleanup_test", 1)

        barrier_key = f"{queue.key_prefix}:barrier:cleanup_test"
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
        """Test concurrent barrier operations with workers.

        BSP model: Multiple workers complete tasks concurrently.
        """
        from graflow.coordination.redis_coordinator import record_task_completion

        coordinator, _, queue = create_coordinator(clean_redis, key_prefix="integration-concurrent")

        # Create barrier for 3 tasks
        coordinator.create_barrier("concurrent_test", 3)

        results = []

        def producer_wait():
            """Producer waits for all workers."""
            result = coordinator.wait_barrier("concurrent_test", timeout=5)
            results.append(("producer", result, time.time()))

        # Start producer waiting
        producer_thread = threading.Thread(target=producer_wait)
        producer_thread.start()

        # Small delay to ensure producer is waiting
        time.sleep(0.1)

        # Simulate 3 workers completing tasks concurrently
        def worker(worker_id):
            time.sleep(0.1)  # Simulate some work
            record_task_completion(
                clean_redis, queue.key_prefix, f"task_{worker_id}", "concurrent_test", True
            )

        worker_threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            worker_threads.append(thread)

        # Wait for all threads
        for thread in worker_threads:
            thread.join()
        producer_thread.join()

        # Producer should succeed
        assert len(results) == 1
        assert results[0][0] == "producer"
        assert results[0][1] is True

        coordinator.cleanup_barrier("concurrent_test")

    def test_barrier_race_condition_fast_workers(self, clean_redis):
        """Test race condition where workers complete before producer waits.

        This tests the fix: producer subscribes first, then checks if already complete.
        """
        from graflow.coordination.redis_coordinator import record_task_completion

        coordinator, _, queue = create_coordinator(clean_redis, key_prefix="integration-race")

        # Create barrier for 2 tasks
        barrier_id = "race_test"
        coordinator.create_barrier(barrier_id, 2)

        # Workers complete IMMEDIATELY (before producer waits)
        record_task_completion(
            clean_redis, queue.key_prefix, "task_1", barrier_id, True
        )
        record_task_completion(
            clean_redis, queue.key_prefix, "task_2", barrier_id, True
        )

        # Now producer waits (should detect completion immediately)
        result = coordinator.wait_barrier(barrier_id, timeout=5)

        # Should return True immediately (not timeout)
        assert result is True

        coordinator.cleanup_barrier(barrier_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
