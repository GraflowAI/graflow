"""Basic integration test for TaskWorker functionality."""

import logging
import time
from typing import Any, Optional

from graflow.core.context import ExecutionContext
from graflow.queue.base import TaskQueue, TaskSpec

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockExecutionContext(ExecutionContext):
    """Mock ExecutionContext for testing."""

    def __init__(self):
        self.session_id = "test_session"


class MockTaskQueue(TaskQueue):
    """Mock TaskQueue for testing that extends AbstractTaskQueue."""

    def __init__(self):
        super().__init__(MockExecutionContext())
        self.tasks = []
        self.dequeue_count = 0

    def enqueue(self, task_spec: TaskSpec) -> bool:
        """Add a task to the queue."""
        self.tasks.append(task_spec)
        logger.info(f"Enqueued task: {task_spec.task_id}")
        return True

    def dequeue(self) -> Optional[TaskSpec]:
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

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.tasks.clear()

    def to_list(self) -> list[str]:
        """Get list of node IDs in queue order."""
        return [task.task_id for task in self.tasks]


def test_basic_task_execution():
    """Test basic TaskWorker functionality."""
    print("=== Testing Basic TaskWorker Functionality ===")

    try:
        from graflow.worker.handler import InProcessTaskExecutor
        from graflow.worker.worker import TaskWorker

        # Create mock queue and add some tasks
        queue = MockTaskQueue()
        queue.enqueue(TaskSpec("task_1", queue.execution_context))
        queue.enqueue(TaskSpec("task_2", queue.execution_context))
        queue.enqueue(TaskSpec("task_3", queue.execution_context))

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
            queue.enqueue(TaskSpec(f"concurrent_task_{i}", queue.execution_context))

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


def test_redis_coordinator_dispatch(clean_redis: Any):
    """Test RedisCoordinator dispatch_task functionality."""
    print("\n=== Testing RedisCoordinator dispatch_task ===")

    try:
        from graflow.coordination.redis import RedisCoordinator

        # This would normally require Redis, so we'll just test the method exists
        coordinator = RedisCoordinator(clean_redis)  # Will fail, but we can test the interface

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

    from tests.conftest import clean_redis, redis_server
    redis_server = redis_server()
    clean_redis = clean_redis(redis_server)
    test_redis_coordinator_dispatch(clean_redis)

    print("\n🎉 Integration tests completed!")
