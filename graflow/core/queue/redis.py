"""Redis distributed task queue implementation."""

import json
from typing import Optional, cast

try:
    import redis
except ImportError:
    redis = None

from .base import AbstractTaskQueue, TaskSpec, TaskStatus


class RedisTaskQueue(AbstractTaskQueue):
    """Redis distributed task queue with TaskSpec support."""

    def __init__(self, execution_context, redis_client: Optional['redis.Redis'] = None,
                 key_prefix: str = "graflow"):
        """Initialize Redis task queue.

        Args:
            execution_context: ExecutionContext instance
            redis_client: Optional Redis client instance
            key_prefix: Key prefix for Redis keys

        Raises:
            ImportError: If redis library is not installed
        """
        if redis is None:
            raise ImportError("Redis library not installed. Install with: pip install redis")

        super().__init__(execution_context)

        self.redis = redis_client or redis.Redis(
            host='localhost', port=6379, db=0, decode_responses=True
        )
        self.key_prefix = key_prefix
        self.session_id = execution_context.session_id

        # Redis keys
        self.queue_key = f"{key_prefix}:queue:{self.session_id}"
        self.specs_key = f"{key_prefix}:specs:{self.session_id}"

    def enqueue(self, task_spec: TaskSpec) -> bool:
        """Add TaskSpec to Redis queue (FIFO)."""
        # Serialize TaskSpec to JSON and store in hash
        spec_data = {
            'node_id': task_spec.node_id,
            'status': task_spec.status.value,
            'created_at': task_spec.created_at,
            # Phase 3: Advanced features
            'retry_count': task_spec.retry_count,
            'max_retries': task_spec.max_retries,
            'last_error': task_spec.last_error
        }
        self.redis.hset(self.specs_key, task_spec.node_id, json.dumps(spec_data))
        self._task_specs[task_spec.node_id] = task_spec

        # Add node ID to queue
        self.redis.rpush(self.queue_key, task_spec.node_id)

        # Phase 3: Metrics
        if self.enable_metrics:
            self.metrics['enqueued'] += 1

        return True

    def dequeue(self) -> Optional[TaskSpec]:
        """Get next TaskSpec from Redis."""
        # Get next node ID from queue
        node_id = self.redis.lpop(self.queue_key)
        if not node_id:
            return None
        node_id = cast(str, node_id)

        # Get TaskSpec from hash and deserialize
        spec_json = self.redis.hget(self.specs_key, node_id)
        if not spec_json:
            return None
        spec_json = cast(str, spec_json)

        spec_data = json.loads(spec_json)
        task_spec = TaskSpec(
            node_id=spec_data['node_id'],
            execution_context=self.execution_context,
            status=TaskStatus(spec_data['status']),
            created_at=spec_data['created_at'],
            # Phase 3: Advanced features
            retry_count=spec_data.get('retry_count', 0),
            max_retries=spec_data.get('max_retries', 3),
            last_error=spec_data.get('last_error')
        )
        task_spec.status = TaskStatus.RUNNING
        self._task_specs[node_id] = task_spec

        # Phase 3: Metrics
        if self.enable_metrics:
            self.metrics['dequeued'] += 1

        return task_spec

    def is_empty(self) -> bool:
        """Check if Redis queue is empty."""
        return self.redis.llen(self.queue_key) == 0

    def size(self) -> int:
        """Get Redis queue size."""
        return cast(int, self.redis.llen(self.queue_key))

    def peek_next_node(self) -> Optional[str]:
        """Peek next node without removing."""
        result = self.redis.lindex(self.queue_key, 0)
        return str(result) if result else None

    def to_list(self) -> list[str]:
        """Get list of node IDs in queue order."""
        queue_items = cast(list, self.redis.lrange(self.queue_key, 0, -1))
        return [str(item) for item in queue_items]

    def cleanup(self) -> None:
        """Clean up Redis keys when session ends."""
        self.redis.delete(self.queue_key, self.specs_key)
