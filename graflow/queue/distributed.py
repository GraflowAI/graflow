"""Redis distributed task queue implementation."""

import json
from typing import Optional, cast

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None

from graflow.coordination.graph_store import GraphStore
from graflow.queue.base import TaskQueue, TaskSpec, TaskStatus


class DistributedTaskQueue(TaskQueue):
    """Redis distributed task queue with TaskSpec support."""

    def __init__(
        self,
        redis_client: Optional["Redis"] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        key_prefix: str = "graflow",
        graph_store: Optional["GraphStore"] = None,
    ):
        """Initialize Redis task queue.

        Args:
            redis_client: Optional Redis client instance
            host: Redis server host
            port: Redis server port
            db: Redis database number
            key_prefix: Key prefix for Redis keys
            graph_store: Optional GraphStore for loading graphs (required for SerializedTaskRecord)

        Raises:
            ImportError: If redis library is not installed
        """
        if redis is None:
            raise ImportError("Redis library not installed. Install with: pip install redis")

        super().__init__()

        if redis_client is not None:
            self.redis_client = redis_client
        else:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

        self.key_prefix = key_prefix
        self.graph_store = graph_store or GraphStore(self.redis_client, key_prefix)

        # Redis keys (prefix-only namespace)
        self.queue_key = f"{key_prefix}:queue"
        # Retain specs_key for backward compatibility with cleanup/tests (unused for new records)
        self.specs_key = f"{key_prefix}:specs"

    def enqueue(self, task_spec: TaskSpec) -> bool:
        """Add TaskSpec to Redis queue (SerializedTaskRecord-based)."""
        from graflow.coordination.records import SerializedTaskRecord

        # Ensure graph_hash exists; save graph snapshot if missing
        graph_hash = getattr(task_spec.execution_context, "graph_hash", None)
        if graph_hash is None:
            graph_hash = self.graph_store.save(task_spec.execution_context.graph)
            task_spec.execution_context.graph_hash = graph_hash

        trace_id = task_spec.trace_id or getattr(task_spec.execution_context, "trace_id", None)
        record = SerializedTaskRecord(
            task_id=task_spec.task_id,
            session_id=task_spec.execution_context.session_id,
            graph_hash=graph_hash,
            trace_id=trace_id or task_spec.execution_context.session_id,
            group_id=getattr(task_spec, "group_id", None),
            parent_span_id=task_spec.parent_span_id,
            created_at=task_spec.created_at,
        )

        self.redis_client.rpush(self.queue_key, record.to_json())
        self._task_specs[task_spec.task_id] = task_spec

        # Phase 3: Metrics
        if self.enable_metrics:
            self.metrics["enqueued"] += 1

        return True

    def dequeue(self) -> Optional[TaskSpec]:
        """Get next TaskSpec from Redis."""
        # Get next item from queue
        item = self.redis_client.lpop(self.queue_key)
        if not item:
            return None

        # Check if it's a SerializedTaskRecord (JSON)
        try:
            # Try to parse as JSON
            data = json.loads(item)  # type: ignore
            if isinstance(data, dict) and "graph_hash" in data:
                # It's a SerializedTaskRecord
                return self._dequeue_record(data)
        except (json.JSONDecodeError, TypeError):
            self._logger.warning(f"Could not decode queue item as SerializedTaskRecord: {item!r}")

        # Unrecognized item
        self._logger.error(f"Dropping unrecognized queue item: {item!r}")
        return None

    def _dequeue_record(self, data: dict) -> Optional[TaskSpec]:
        """Handle SerializedTaskRecord."""
        if not self.graph_store:
            # If graph_store is missing, we can't process this record
            # Should we re-enqueue? Or log error?
            # For now, log error and return None (task lost)
            self._logger.error(
                "GraphStore not configured in DistributedTaskQueue, cannot process SerializedTaskRecord",
                extra={"queue_key": self.queue_key, "task_data_keys": list(data.keys())},
            )
            return None

        from graflow.coordination.records import SerializedTaskRecord
        from graflow.worker.context_factory import ExecutionContextFactory

        try:
            record = SerializedTaskRecord(**data)
            context, task = ExecutionContextFactory.create_from_record(record, self.graph_store)

            task_spec = TaskSpec(
                executable=task, execution_context=context, status=TaskStatus.RUNNING, created_at=record.created_at
            )

            if record.group_id:
                task_spec.group_id = record.group_id
            task_spec.trace_id = record.trace_id
            task_spec.parent_span_id = record.parent_span_id

            self._task_specs[record.task_id] = task_spec

            if self.enable_metrics:
                self.metrics["dequeued"] += 1

            return task_spec

        except Exception as e:
            self._logger.error(
                "Error processing SerializedTaskRecord: %s",
                str(e),
                exc_info=True,
                extra={"queue_key": self.queue_key, "record_data": data},
            )
            return None

    def is_empty(self) -> bool:
        """Check if Redis queue is empty."""
        return self.redis_client.llen(self.queue_key) == 0

    def size(self) -> int:
        """Get Redis queue size."""
        return cast(int, self.redis_client.llen(self.queue_key))

    def notify_task_completion(
        self,
        task_id: str,
        success: bool,
        group_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Notify task completion to redis.py functions.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            group_id: Group ID for barrier synchronization
            error_message: Error message if task failed
        """
        if group_id:
            from graflow.coordination.redis_coordinator import record_task_completion

            record_task_completion(
                self.redis_client,
                self.key_prefix,
                task_id,
                group_id,
                success,
                error_message,
            )

    def cleanup(self) -> None:
        """Clean up Redis keys when session ends."""
        self.redis_client.delete(self.queue_key, self.specs_key)
