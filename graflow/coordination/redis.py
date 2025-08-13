"""Redis-based coordination backend for distributed parallel execution."""

import threading
import time
from typing import Any, Callable, Dict

from graflow.coordination.coordinator import TaskCoordinator
from graflow.coordination.task_spec import TaskSpec


class RedisCoordinator(TaskCoordinator):
    """Redis-based task coordination for distributed execution."""

    def __init__(self, redis_client, task_queue):
        """Initialize Redis coordinator with Redis client and task queue.

        Args:
            redis_client: Redis client instance
            task_queue: RedisTaskQueue instance for task dispatch
        """
        self.redis = redis_client
        self.task_queue = task_queue
        self.active_barriers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

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
        # Use RedisTaskQueue's enqueue method
        self.task_queue.enqueue(task_spec)

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
