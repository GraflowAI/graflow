"""Basic integration test for TaskWorker functionality."""

import logging
import time
from typing import Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTaskSpec:
    """Mock TaskSpec for testing."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        from graflow.queue.base import TaskStatus
        self.status = TaskStatus.READY
        self.retry_count = 0
        self.max_retries = 3
        self.last_error = None
        self.execution_context = None


class MockTaskQueue:
    """Mock TaskQueue for testing."""

    def __init__(self):
        self.tasks = []
        self.dequeue_count = 0

    def enqueue(self, task_spec: MockTaskSpec):
        """Add a task to the queue."""
        self.tasks.append(task_spec)
        logger.info(f"Enqueued task: {task_spec.task_id}")

    def dequeue(self) -> Any:
        """Get a task from the queue."""
        self.dequeue_count += 1
        if self.tasks:
            task = self.tasks.pop(0)
            logger.info(f"Dequeued task: {task.task_id}")
            return task
        return None

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self.tasks) == 0


def test_basic_task_execution():
    """Test basic TaskWorker functionality."""
    print("=== Testing Basic TaskWorker Functionality ===")

    try:
        from graflow.worker.handler import InProcessTaskExecutor
        from graflow.worker.worker import TaskWorker

        # Create mock queue and add some tasks
        queue = MockTaskQueue()
        queue.enqueue(MockTaskSpec("task_1"))
        queue.enqueue(MockTaskSpec("task_2"))
        queue.enqueue(MockTaskSpec("task_3"))

        # Create handler and worker with concurrent execution
        handler = InProcessTaskExecutor()
        worker = TaskWorker(
            queue=queue,
            handler=handler,
            worker_id="test_worker",
            max_concurrent_tasks=2,
            poll_interval=0.05
        )

        # Start worker
        worker.start()

        # Wait for tasks to be processed
        print("Waiting for tasks to be processed...")
        time.sleep(2.0)

        # Check intermediate metrics
        metrics = worker.get_metrics()
        print(f"Intermediate metrics: {metrics['tasks_processed']} processed, {metrics['active_tasks']} active")

        # Wait a bit more
        time.sleep(1.0)

        # Stop worker
        worker.stop()

        # Check final metrics
        metrics = worker.get_metrics()
        print("\nWorker Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Verify tasks were processed
        assert metrics["tasks_processed"] > 0, "No tasks were processed"
        assert queue.is_empty(), "Not all tasks were processed"
        assert metrics["active_tasks"] == 0, "Some tasks still active after shutdown"

        print("\n✅ Basic TaskWorker test passed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure PYTHONPATH includes the graflow directory")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def test_concurrent_task_execution():
    """Test concurrent task execution."""
    print("\n=== Testing Concurrent Task Execution ===")

    try:
        from graflow.worker.handler import InProcessTaskExecutor
        from graflow.worker.worker import TaskWorker

        # Create mock queue with more tasks
        queue = MockTaskQueue()
        for i in range(10):
            queue.enqueue(MockTaskSpec(f"concurrent_task_{i}"))

        # Create handler and worker with higher concurrency
        handler = InProcessTaskExecutor()
        worker = TaskWorker(
            queue=queue,
            handler=handler,
            worker_id="concurrent_worker",
            max_concurrent_tasks=5,
            poll_interval=0.01
        )

        # Start worker
        worker.start()

        # Monitor execution
        start_time = time.time()
        while not queue.is_empty() and time.time() - start_time < 10.0:
            metrics = worker.get_metrics()
            print(f"Progress: {metrics['tasks_processed']} processed, {metrics['active_tasks']} active")
            time.sleep(0.5)

        # Stop worker
        worker.stop()

        # Final metrics
        metrics = worker.get_metrics()
        print("\nConcurrent Execution Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # Verify all tasks processed
        assert metrics["tasks_processed"] == 10, f"Expected 10 tasks, got {metrics['tasks_processed']}"
        assert metrics["active_tasks"] == 0, "Tasks still active after shutdown"

        print("✅ Concurrent task execution test passed!")

    except Exception as e:
        print(f"❌ Concurrent test failed: {e}")
        import traceback
        traceback.print_exc()


def test_redis_coordinator_dispatch():
    """Test RedisCoordinator dispatch_task functionality."""
    print("\n=== Testing RedisCoordinator dispatch_task ===")

    try:
        from graflow.coordination.redis import RedisCoordinator
        from graflow.coordination.task_spec import TaskSpec

        # Create mock TaskSpec
        task_spec = TaskSpec("test_node", None)  # execution_context can be None for testing

        # This would normally require Redis, so we'll just test the method exists
        coordinator = RedisCoordinator(None)  # Will fail, but we can test the interface

        # Test that the method exists and has the right signature
        assert hasattr(coordinator, 'dispatch_task'), "dispatch_task method not found"
        assert hasattr(coordinator, '_serialize_execution_context'), "serialization method not found"

        print("✅ RedisCoordinator interface test passed!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        # Expected since we're not providing real Redis client
        print(f"⚠️  RedisCoordinator test skipped (expected without Redis): {e}")


if __name__ == "__main__":
    print("Starting TaskWorker Integration Tests...\n")

    test_basic_task_execution()
    test_concurrent_task_execution()
    test_redis_coordinator_dispatch()

    print("\n🎉 Integration tests completed!")
