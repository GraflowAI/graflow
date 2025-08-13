"""Redis-based coordination backend for distributed parallel execution."""

import threading
import time
from typing import Any, Callable, Dict, List

from graflow.coordination.coordinator import TaskCoordinator
from graflow.coordination.task_spec import TaskSpec
from graflow.queue.base import TaskSpec as QueueTaskSpec
from graflow.queue.redis import RedisTaskQueue


class RedisCoordinator(TaskCoordinator):
    """Redis-based task coordination for distributed execution."""

    def __init__(self, task_queue: RedisTaskQueue):
        """Initialize Redis coordinator with RedisTaskQueue.

        Args:
            task_queue: RedisTaskQueue instance for task dispatch and barrier coordination
        """
        self.task_queue = task_queue
        self.redis = task_queue.redis  # Use Redis client from task queue
        self.active_barriers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def execute_group(self, group_id: str, tasks: List[TaskSpec]) -> None:
        """Execute parallel group with barrier synchronization."""
        barrier_id = self.create_barrier(group_id, len(tasks))
        try:
            for task in tasks:
                self.dispatch_task(task, group_id)
            if not self.wait_barrier(barrier_id):
                raise TimeoutError(f"Barrier wait timeout for group {group_id}")
            print(f"Parallel group {group_id} completed successfully")
        finally:
            self.cleanup_barrier(barrier_id)

    def create_barrier(self, barrier_id: str, participant_count: int) -> str:
        """Create a barrier for parallel task synchronization."""
        barrier_key = f"barrier:{barrier_id}"
        completion_channel = f"barrier_done:{barrier_id}"

        with self._lock:
            # Reset barrier state
            self.redis.delete(barrier_key)
            self.redis.set(f"{barrier_key}:expected", participant_count)

            self.active_barriers[barrier_id] = {
                "key": barrier_key,
                "channel": completion_channel,
                "expected": participant_count,
                "current": 0
            }

        return barrier_key

    def wait_barrier(self, barrier_id: str, timeout: int = 30) -> bool:
        """Wait at barrier until all participants arrive."""
        if barrier_id not in self.active_barriers:
            return False

        barrier_info = self.active_barriers[barrier_id]

        # Increment participant count atomically
        current_count = self.redis.incr(barrier_info["key"])

        if current_count >= barrier_info["expected"]:
            # Last participant - notify all waiting participants
            self.redis.publish(barrier_info["channel"], "complete")
            return True
        else:
            # Wait for completion notification
            pubsub = self.redis.pubsub()
            pubsub.subscribe(barrier_info["channel"])

            start_time = time.time()
            try:
                for message in pubsub.listen():
                    if message["type"] == "message" and message["data"] == b"complete":
                        return True

                    # Check timeout
                    if time.time() - start_time > timeout:
                        return False

            except Exception:
                return False
            finally:
                pubsub.close()

        return False

    def signal_barrier(self, barrier_id: str) -> None:
        """Signal barrier completion."""
        if barrier_id in self.active_barriers:
            barrier_info = self.active_barriers[barrier_id]
            current_count = self.redis.incr(barrier_info["key"])

            if current_count >= barrier_info["expected"]:
                self.redis.publish(barrier_info["channel"], "complete")

    def dispatch_task(self, task_spec: TaskSpec, group_id: str) -> None:
        """Dispatch task to Redis queue for worker processing."""
        # Convert coordination TaskSpec to queue TaskSpec format
        queue_task_spec = self._convert_to_queue_task_spec(task_spec)

        # Use RedisTaskQueue's enqueue method
        self.task_queue.enqueue(queue_task_spec)

    def _convert_to_queue_task_spec(self, task_spec: TaskSpec) -> QueueTaskSpec:
        """Convert coordination TaskSpec to queue TaskSpec format."""
        return QueueTaskSpec(
            task_id=task_spec.task_id,
            execution_context=task_spec.execution_context,
        )

    def cleanup_barrier(self, barrier_id: str) -> None:
        """Clean up barrier resources."""
        if barrier_id in self.active_barriers:
            barrier_info = self.active_barriers[barrier_id]

            # Clean up Redis keys
            self.redis.delete(barrier_info["key"])
            self.redis.delete(f"{barrier_info['key']}:expected")

            # Remove from active barriers
            with self._lock:
                del self.active_barriers[barrier_id]

    def get_task_registry(self) -> Dict[str, Callable]:
        """Get registry of available task functions for Redis workers."""
        # This would be populated with registered task functions
        # In a real implementation, this would be managed by the workflow engine
        return getattr(self, '_task_registry', {})

    def register_task_function(self, name: str, func: Callable) -> None:
        """Register a task function for Redis workers."""
        if not hasattr(self, '_task_registry'):
            self._task_registry = {}
        self._task_registry[name] = func

    def get_queue_size(self, group_id: str) -> int:
        """Get current queue size for a group."""
        return self.task_queue.size()

    def clear_queue(self, group_id: str) -> None:
        """Clear all tasks from a group's queue."""
        self.task_queue.cleanup()
