"""Redis-based coordination backend for distributed parallel execution."""

from __future__ import annotations

import json
import logging
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

logger = logging.getLogger(__name__)


class RedisCoordinator(TaskCoordinator):
    """Redis-based task coordination for distributed execution."""

    def __init__(self, task_queue: DistributedTaskQueue):
        """Initialize Redis coordinator with RedisTaskQueue.

        Args:
            task_queue: RedisTaskQueue instance for task dispatch and barrier coordination
        """
        import threading  # Import here to avoid conflict with local threading.py

        logger.info("Initializing RedisCoordinator with queue key_prefix=%s", task_queue.key_prefix)
        self.task_queue = task_queue
        self.redis = task_queue.redis_client  # Use Redis client from task queue
        self.active_barriers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Initialize GraphStore (reuse queue's instance when available)
        if self.task_queue.graph_store:
            logger.debug("Reusing existing GraphStore from task queue")
            self.graph_store = self.task_queue.graph_store
        else:
            logger.debug("Creating new GraphStore instance")
            self.graph_store = GraphStore(self.redis, self.task_queue.key_prefix)
            self.task_queue.graph_store = self.graph_store

        logger.info("RedisCoordinator initialized successfully")

    def execute_group(
        self, group_id: str, tasks: List[Executable], execution_context: ExecutionContext, policy: GroupExecutionPolicy
    ) -> None:
        """Execute parallel group with barrier synchronization.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            execution_context: Execution context
            policy: GroupExecutionPolicy instance for result evaluation
        """
        from graflow.core.handler import TaskResult

        logger.info("Executing parallel group '%s' with %d tasks", group_id, len(tasks))
        logger.debug("Tasks in group '%s': %s", group_id, [t.task_id for t in tasks])

        # Lazy Upload: Save current graph state
        logger.debug("Saving graph state for group '%s'", group_id)
        graph_hash = self.graph_store.save(execution_context.graph)
        execution_context.graph_hash = graph_hash
        logger.debug("Graph saved with hash: %s", graph_hash[:8])

        self.create_barrier(group_id, len(tasks))
        try:
            logger.info("Dispatching %d tasks to Redis queue", len(tasks))
            for task in tasks:
                self.dispatch_task(task, group_id)
            logger.info("All tasks dispatched for group '%s', waiting for completion", group_id)

            if not self.wait_barrier(group_id):
                logger.error("Barrier wait timeout for group '%s'", group_id)
                raise TimeoutError(f"Barrier wait timeout for group {group_id}")

            # Collect results from Redis
            logger.info("Collecting completion results for group '%s'", group_id)
            completion_results = self._get_completion_results(group_id)
            logger.info("Retrieved %d completion results", len(completion_results))

            # Convert to TaskResult format and check for graph updates
            task_results: Dict[str, TaskResult] = {}
            failed_tasks = []

            for result_data in completion_results:
                task_id = result_data["task_id"]
                success = result_data["success"]
                task_results[task_id] = TaskResult(
                    task_id=task_id,
                    success=success,
                    error_message=result_data.get("error_message"),
                    timestamp=result_data.get("timestamp", 0.0),
                )
                if not success:
                    failed_tasks.append(task_id)

            if failed_tasks:
                logger.warning("Group '%s' completed with failures: %s", group_id, failed_tasks)
            else:
                logger.info("Parallel group '%s' completed successfully", group_id)

            logger.info("Parallel group '%s' completed", group_id)

            # Apply policy's group execution logic directly
            logger.debug("Applying group policy for '%s'", group_id)
            policy.on_group_finished(group_id, tasks, task_results, execution_context)
        except Exception as e:
            logger.error("Error executing group '%s': %s", group_id, e, exc_info=True)
            raise
        finally:
            logger.debug("Cleaning up barrier for group '%s'", group_id)
            self.cleanup_barrier(group_id)

    def create_barrier(self, barrier_id: str, participant_count: int) -> str:
        """Create a barrier for parallel task synchronization."""
        barrier_key = f"{self.task_queue.key_prefix}:barrier:{barrier_id}"
        completion_channel = f"{self.task_queue.key_prefix}:barrier_done:{barrier_id}"

        logger.debug("Creating barrier '%s' for %d participants", barrier_id, participant_count)

        with self._lock:
            # Reset barrier state
            self.redis.delete(barrier_key)
            self.redis.set(f"{barrier_key}:expected", participant_count)

            self.active_barriers[barrier_id] = {
                "key": barrier_key,
                "channel": completion_channel,
                "expected": participant_count,
                "current": 0,
            }

        logger.debug("Barrier '%s' created successfully (channel: %s)", barrier_id, completion_channel)
        return barrier_key

    def wait_barrier(self, barrier_id: str, timeout: int = 30) -> bool:
        """Wait at barrier until all participants arrive.

        Producer only subscribes and waits - workers increment the barrier counter.
        This implements the BSP (Bulk Synchronous Parallel) model where the producer
        dispatches all tasks and waits, while workers execute and signal completion.

        Uses exponential backoff (1s, 2s, 4s, 8s, 16s, 32s, 60s) to avoid busy polling
        and reduce Redis load.
        """
        if barrier_id not in self.active_barriers:
            logger.warning("Barrier '%s' not found in active barriers", barrier_id)
            return False

        barrier_info = self.active_barriers[barrier_id]
        logger.info(
            "Waiting for barrier '%s' (expected: %d, timeout: %ds)", barrier_id, barrier_info["expected"], timeout
        )

        # Subscribe to completion channel FIRST (before checking counter)
        # This prevents race condition where workers complete before we subscribe
        pubsub = self.redis.pubsub()
        pubsub.subscribe(barrier_info["channel"])
        logger.debug("Subscribed to completion channel: %s", barrier_info["channel"])

        try:
            # Wait for completion notification with exponential backoff
            logger.debug("Listening for completion messages on barrier '%s'", barrier_id)

            # Exponential backoff parameters (1s, 2s, 4s, 8s, 16s, 32s, 60s)
            min_backoff = 1.0  # 1s minimum
            max_backoff = 60.0  # 60s maximum
            current_backoff = min_backoff

            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    logger.error(
                        "Barrier '%s' timeout after %.2fs (expected %d tasks)",
                        barrier_id,
                        elapsed,
                        barrier_info["expected"],
                    )
                    return False

                # Non-blocking polling with exponential backoff
                message = pubsub.get_message(timeout=current_backoff)
                if message and message.get("type") == "message":
                    data = message.get("data")
                    if data in ("complete", b"complete"):
                        logger.info("Barrier '%s' completed after %.2fs", barrier_id, elapsed)
                        return True

                # Fallback: check counter in case the publish was missed
                current_count_bytes = self.redis.get(barrier_info["key"])
                if current_count_bytes:
                    current_count = int(current_count_bytes)  # type: ignore[arg-type]
                    if current_count >= barrier_info["expected"]:
                        logger.info(
                            "Barrier '%s' completed via counter check after %.2fs (%d/%d tasks)",
                            barrier_id,
                            elapsed,
                            current_count,
                            barrier_info["expected"],
                        )
                        return True

                # Increase backoff exponentially with upper bound
                current_backoff = min(current_backoff * 2, max_backoff)

        except Exception as e:
            logger.error("Exception in wait_barrier for '%s': %s", barrier_id, e, exc_info=True)
            return False
        finally:
            pubsub.close()
            logger.debug("Closed pubsub connection for barrier '%s'", barrier_id)

        return False

    def dispatch_task(self, executable: Executable, group_id: str) -> None:
        """Dispatch task to Redis queue for worker processing."""
        logger.debug("Dispatching task '%s' to group '%s'", executable.task_id, group_id)
        context = executable.get_execution_context()

        # graph_hash is set in execution_context by execute_group
        graph_hash = getattr(context, "graph_hash", None)
        if graph_hash is None:
            logger.error("graph_hash not set in ExecutionContext for task '%s'", executable.task_id)
            raise ValueError("graph_hash not set in ExecutionContext")

        record = SerializedTaskRecord(
            task_id=executable.task_id,
            session_id=context.session_id,
            graph_hash=graph_hash,
            trace_id=context.trace_id,
            parent_span_id=context.tracer.get_current_span_id() if context.tracer else None,
            group_id=group_id,
            created_at=time.time(),
        )

        logger.debug(
            "Pushing task '%s' to queue (session: %s, graph: %s)",
            executable.task_id,
            context.session_id,
            graph_hash[:8],
        )

        # Push directly to Redis queue bypassing RedisTaskQueue.enqueue()
        self.task_queue.redis_client.lpush(self.task_queue.queue_key, record.to_json())
        logger.debug("Task '%s' successfully dispatched to Redis queue", executable.task_id)

    def _get_completion_results(self, group_id: str) -> List[Dict[str, Any]]:
        """Retrieve all completion records from Redis.

        Args:
            group_id: Parallel group identifier

        Returns:
            List of completion result dictionaries
        """
        completion_key = f"{self.task_queue.key_prefix}:completions:{group_id}"
        logger.debug("Retrieving completion results for group '%s' from key: %s", group_id, completion_key)
        records = self.redis.hgetall(completion_key)

        results = []
        # Type ignore: redis-py hgetall returns dict synchronously, not Awaitable
        for task_id, record_json in records.items():  # type: ignore[union-attr]
            try:
                data = json.loads(record_json)
                # Ensure task_id is in the data
                data["task_id"] = task_id.decode() if isinstance(task_id, bytes) else task_id
                results.append(data)
                logger.debug(
                    "Retrieved completion record for task '%s' (success: %s)",
                    data["task_id"],
                    data.get("success", "unknown"),
                )
            except json.JSONDecodeError:
                logger.warning("Failed to decode completion record for task_id=%s in group '%s'", task_id, group_id)
                continue

        logger.debug("Retrieved %d completion results for group '%s'", len(results), group_id)
        return results

    def cleanup_barrier(self, barrier_id: str) -> None:
        """Clean up barrier resources."""
        logger.debug("Cleaning up barrier '%s'", barrier_id)
        if barrier_id in self.active_barriers:
            barrier_info = self.active_barriers[barrier_id]

            # Clean up Redis keys
            logger.debug("Deleting Redis keys for barrier '%s'", barrier_id)
            self.redis.delete(barrier_info["key"])
            self.redis.delete(f"{barrier_info['key']}:expected")
            self.redis.delete(f"{self.task_queue.key_prefix}:completions:{barrier_id}")

            # Remove from active barriers
            with self._lock:
                del self.active_barriers[barrier_id]
            logger.debug("Barrier '%s' cleanup completed", barrier_id)
        else:
            logger.warning("Attempted to cleanup non-existent barrier '%s'", barrier_id)

    def get_queue_size(self, _group_id: str) -> int:
        """Get current queue size for a group."""
        return self.task_queue.size()

    def clear_queue(self, _group_id: str) -> None:
        """Clear all tasks from a group's queue."""
        self.task_queue.cleanup()


def record_task_completion(
    redis_client, key_prefix: str, task_id: str, group_id: str, success: bool, error_message: Optional[str] = None
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
    logger.debug("Recording completion for task '%s' in group '%s' (success: %s)", task_id, group_id, success)
    completion_key = f"{key_prefix}:completions:{group_id}"

    completion_data = {"task_id": task_id, "success": success, "timestamp": time.time(), "error_message": error_message}

    # Store in hash with task_id as key (prevents duplicates/overwrites)
    redis_client.hset(completion_key, task_id, json.dumps(completion_data))
    logger.debug("Stored completion record for task '%s' at key: %s", task_id, completion_key)

    # Trigger barrier signaling using existing pub/sub mechanism
    barrier_key = f"{key_prefix}:barrier:{group_id}"
    current_count = redis_client.incr(barrier_key)
    logger.debug("Incremented barrier count for group '%s': %d", group_id, current_count)

    # Check if barrier is complete
    expected_key = f"{barrier_key}:expected"
    expected_count = redis_client.get(expected_key)

    if not expected_count:
        logger.warning("No expected count found for barrier '%s' (key: %s)", group_id, expected_key)
        return

    expected_int = int(expected_count)
    logger.debug("Barrier '%s' progress: %d/%d tasks completed", group_id, current_count, expected_int)

    if current_count >= expected_int:
        # All tasks completed - publish barrier completion
        completion_channel = f"{key_prefix}:barrier_done:{group_id}"
        logger.info(
            "All tasks completed for group '%s' (%d/%d), publishing completion signal",
            group_id,
            current_count,
            expected_int,
        )
        result = redis_client.publish(completion_channel, "complete")
        logger.debug("Published barrier completion to channel '%s' (subscribers: %d)", completion_channel, result)
