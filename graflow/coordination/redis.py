"""Redis-based coordination backend for distributed parallel execution."""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graflow.coordination.coordinator import TaskCoordinator
from graflow.coordination.graph_store import GraphStore
from graflow.coordination.records import SerializedTaskRecord
from graflow.queue.distributed import DistributedTaskQueue

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.handlers.group_policy import GroupExecutionPolicy
    from graflow.core.task import Executable


class RedisCoordinator(TaskCoordinator):
    """Redis-based task coordination for distributed execution."""

    def __init__(self, task_queue: DistributedTaskQueue):
        """Initialize Redis coordinator with RedisTaskQueue.

        Args:
            task_queue: RedisTaskQueue instance for task dispatch and barrier coordination
        """
        self.task_queue = task_queue
        self.redis = task_queue.redis_client  # Use Redis client from task queue
        self.active_barriers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Initialize GraphStore (reuse queue's instance when available)
        if self.task_queue.graph_store:
            self.graph_store = self.task_queue.graph_store
        else:
            self.graph_store = GraphStore(
                self.redis,
                self.task_queue.key_prefix
            )
            self.task_queue.graph_store = self.graph_store

    def execute_group(
        self,
        group_id: str,
        tasks: List[Executable],
        execution_context: ExecutionContext,
        policy: GroupExecutionPolicy
    ) -> None:
        """Execute parallel group with barrier synchronization.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            execution_context: Execution context
            policy: GroupExecutionPolicy instance for result evaluation
        """
        from graflow.core.handler import TaskResult

        # Lazy Upload: Save current graph state
        graph_hash = self.graph_store.save(execution_context.graph)
        execution_context.graph_hash = graph_hash

        self.create_barrier(group_id, len(tasks))
        try:
            for task in tasks:
                self.dispatch_task(task, group_id)

            if not self.wait_barrier(group_id):
                raise TimeoutError(f"Barrier wait timeout for group {group_id}")

            # Collect results from Redis
            completion_results = self._get_completion_results(group_id)

            # Convert to TaskResult format and check for graph updates
            task_results: Dict[str, TaskResult] = {}

            for result_data in completion_results:
                task_id = result_data["task_id"]
                task_results[task_id] = TaskResult(
                    task_id=task_id,
                    success=result_data["success"],
                    error_message=result_data.get("error_message"),
                    timestamp=result_data.get("timestamp", 0.0)
                )

            print(f"Parallel group {group_id} completed")

            # Apply policy's group execution logic directly
            policy.on_group_finished(group_id, tasks, task_results, execution_context)
        finally:
            self.cleanup_barrier(group_id)

    def create_barrier(self, barrier_id: str, participant_count: int) -> str:
        """Create a barrier for parallel task synchronization."""
        barrier_key = f"{self.task_queue.key_prefix}:barrier:{barrier_id}"
        completion_channel = f"{self.task_queue.key_prefix}:barrier_done:{barrier_id}"

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
        """Wait at barrier until all participants arrive.

        Producer only subscribes and waits - workers increment the barrier counter.
        This implements the BSP (Bulk Synchronous Parallel) model where the producer
        dispatches all tasks and waits, while workers execute and signal completion.
        """
        if barrier_id not in self.active_barriers:
            return False

        barrier_info = self.active_barriers[barrier_id]

        # Subscribe to completion channel FIRST (before checking counter)
        # This prevents race condition where workers complete before we subscribe
        pubsub = self.redis.pubsub()
        pubsub.subscribe(barrier_info["channel"])

        try:
            # After subscribing, check if barrier already complete
            # This handles the case where all workers finished before we subscribed
            current_count_bytes = self.redis.get(barrier_info["key"])
            if current_count_bytes:
                current_count = int(current_count_bytes)  # type: ignore[arg-type]
                if current_count >= barrier_info["expected"]:
                    return True

            # Wait for completion notification
            start_time = time.time()
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

    def dispatch_task(self, executable: Executable, group_id: str) -> None:
        """Dispatch task to Redis queue for worker processing."""
        context = executable.get_execution_context()

        # graph_hash is set in execution_context by execute_group
        graph_hash = getattr(context, 'graph_hash', None)
        if graph_hash is None:
            raise ValueError("graph_hash not set in ExecutionContext")

        record = SerializedTaskRecord(
            task_id=executable.task_id,
            session_id=context.session_id,
            graph_hash=graph_hash,
            trace_id=context.trace_id,
            parent_span_id=context.tracer.get_current_span_id() if context.tracer else None,
            group_id=group_id,
            created_at=time.time()
        )

        # Push directly to Redis queue bypassing RedisTaskQueue.enqueue()
        self.task_queue.redis_client.lpush(
            self.task_queue.queue_key,
            record.to_json()
        )

    def _get_completion_results(self, group_id: str) -> List[Dict[str, Any]]:
        """Retrieve all completion records from Redis.

        Args:
            group_id: Parallel group identifier

        Returns:
            List of completion result dictionaries
        """
        completion_key = f"{self.task_queue.key_prefix}:completions:{group_id}"
        records = self.redis.hgetall(completion_key)

        results = []
        # Type ignore: redis-py hgetall returns dict synchronously, not Awaitable
        for task_id, record_json in records.items():  # type: ignore[union-attr]
            try:
                data = json.loads(record_json)
                # Ensure task_id is in the data
                data["task_id"] = task_id.decode() if isinstance(task_id, bytes) else task_id
                results.append(data)
            except json.JSONDecodeError:
                continue

        return results

    def cleanup_barrier(self, barrier_id: str) -> None:
        """Clean up barrier resources."""
        if barrier_id in self.active_barriers:
            barrier_info = self.active_barriers[barrier_id]

            # Clean up Redis keys
            self.redis.delete(barrier_info["key"])
            self.redis.delete(f"{barrier_info['key']}:expected")
            self.redis.delete(f"{self.task_queue.key_prefix}:completions:{barrier_id}")

            # Remove from active barriers
            with self._lock:
                del self.active_barriers[barrier_id]

    def get_queue_size(self, _group_id: str) -> int:
        """Get current queue size for a group."""
        return self.task_queue.size()

    def clear_queue(self, _group_id: str) -> None:
        """Clear all tasks from a group's queue."""
        self.task_queue.cleanup()


def record_task_completion(
    redis_client,
    key_prefix: str,
    task_id: str,
    group_id: str,
    success: bool,
    error_message: Optional[str] = None
):
    """Record task completion in Redis for barrier tracking.

    Args:
        redis_client: Redis client instance
        key_prefix: Key prefix for Redis keys
        task_id: Task identifier
        group_id: Group identifier
        success: Whether task succeeded
        error_message: Error message if task failed

    Note:
        Task result values are stored in ExecutionContext, not in completion records.
        This keeps completion records lightweight and focused on execution status.
    """
    completion_key = f"{key_prefix}:completions:{group_id}"

    completion_data = {
        "task_id": task_id,
        "success": success,
        "timestamp": time.time(),
        "error_message": error_message
    }

    # Store in hash with task_id as key (prevents duplicates/overwrites)
    redis_client.hset(
        completion_key,
        task_id,
        json.dumps(completion_data)
    )

    # Trigger barrier signaling using existing pub/sub mechanism
    barrier_key = f"{key_prefix}:barrier:{group_id}"
    current_count = redis_client.incr(barrier_key)

    # Check if barrier is complete
    expected_key = f"{barrier_key}:expected"
    expected_count = redis_client.get(expected_key)

    if not expected_count:
        return

    if current_count >= int(expected_count):
        # All tasks completed - publish barrier completion
        completion_channel = f"{key_prefix}:barrier_done:{group_id}"
        redis_client.publish(completion_channel, "complete")
