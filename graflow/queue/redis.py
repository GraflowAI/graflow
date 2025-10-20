"""Redis distributed task queue implementation."""

import json
from typing import Optional, cast

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None

from graflow.queue.base import TaskQueue, TaskSpec, TaskStatus


class RedisTaskQueue(TaskQueue):
    """Redis distributed task queue with TaskSpec support."""

    def __init__(self, execution_context, redis_client: Optional['Redis'] = None,
                 host: str = "localhost", port: int = 6379, db: int = 0,
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

        if redis_client is not None:
            self.redis_client = redis_client
        else:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

        self.key_prefix = key_prefix
        self.session_id = execution_context.session_id

        # Redis keys
        self.queue_key = f"{key_prefix}:queue:{self.session_id}"
        self.specs_key = f"{key_prefix}:specs:{self.session_id}"

    def enqueue(self, task_spec: TaskSpec) -> bool:
        """Add TaskSpec to Redis queue (FIFO)."""
        # Serialize TaskSpec to JSON and store in hash
        try:
            task_data = task_spec.task_data
        except ValueError:
            # If task can't be serialized, store None
            task_data = None

        spec_data = {
            'task_id': task_spec.task_id,
            'status': task_spec.status.value,
            'created_at': task_spec.created_at,
            'strategy': task_spec.strategy,
            'task_data': task_data,
            # Phase 3: Advanced features
            'retry_count': task_spec.retry_count,
            'max_retries': task_spec.max_retries,
            'last_error': task_spec.last_error,
            # Phase 2: group_id support
            'group_id': getattr(task_spec, 'group_id', None)
        }
        self.redis_client.hset(self.specs_key, task_spec.task_id, json.dumps(spec_data))
        self._task_specs[task_spec.task_id] = task_spec

        # Add node ID to queue
        self.redis_client.rpush(self.queue_key, task_spec.task_id)

        # Phase 3: Metrics
        if self.enable_metrics:
            self.metrics['enqueued'] += 1

        return True

    def dequeue(self) -> Optional[TaskSpec]:
        """Get next TaskSpec from Redis."""
        # Get next node ID from queue
        task_id = self.redis_client.lpop(self.queue_key)
        if not task_id:
            return None
        task_id = cast(str, task_id)

        # Get TaskSpec from hash and deserialize
        spec_json = self.redis_client.hget(self.specs_key, task_id)
        if not spec_json:
            return None
        spec_json = cast(str, spec_json)

        spec_data = json.loads(spec_json)

        # Try to reconstruct the executable with proper task data
        task_id_from_spec = spec_data.get('task_id', spec_data.get('node_id', task_id))
        task_data = spec_data.get('task_data')

        if task_data:
            # Deserialize the task and create a proper executable
            try:
                func = self.execution_context.task_resolver.resolve_task(task_data)

                # Create a TaskWrapper with the resolved function
                from graflow.core.task import TaskWrapper
                placeholder_task = TaskWrapper(task_id_from_spec, func, register_to_context=False)

            except Exception:
                # Fall back to placeholder if deserialization fails
                from graflow.core.task import Task
                placeholder_task = Task(task_id_from_spec, register_to_context=False)
        else:
            # No function data available, use placeholder
            from graflow.core.task import Task
            placeholder_task = Task(task_id_from_spec, register_to_context=False)

        task_spec = TaskSpec(
            executable=placeholder_task,
            execution_context=self.execution_context,
            strategy=spec_data.get('strategy', 'reference'),
            status=TaskStatus(spec_data['status']),
            created_at=spec_data['created_at'],
            # Phase 3: Advanced features
            retry_count=spec_data.get('retry_count', 0),
            max_retries=spec_data.get('max_retries', 3),
            last_error=spec_data.get('last_error')
        )
        # Phase 2: Restore group_id
        group_id = spec_data.get('group_id')
        if group_id:
            task_spec.group_id = group_id
        task_spec.status = TaskStatus.RUNNING
        self._task_specs[task_id] = task_spec

        # Phase 3: Metrics
        if self.enable_metrics:
            self.metrics['dequeued'] += 1

        return task_spec

    def is_empty(self) -> bool:
        """Check if Redis queue is empty."""
        return self.redis_client.llen(self.queue_key) == 0

    def size(self) -> int:
        """Get Redis queue size."""
        return cast(int, self.redis_client.llen(self.queue_key))

    def peek_next_node(self) -> Optional[str]:
        """Peek next node without removing."""
        result = self.redis_client.lindex(self.queue_key, 0)
        return str(result) if result else None

    def to_list(self) -> list[str]:
        """Get list of node IDs in queue order."""
        queue_items = cast(list, self.redis_client.lrange(self.queue_key, 0, -1))
        return [str(item) for item in queue_items]

    def notify_task_completion(
        self,
        task_id: str,
        success: bool,
        group_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Notify task completion to redis.py functions.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            group_id: Group ID for barrier synchronization
            error_message: Error message if task failed
        """
        if group_id:
            from graflow.coordination.redis import record_task_completion
            record_task_completion(
                self.redis_client, self.key_prefix,
                task_id, group_id, success, error_message
            )

    def cleanup(self) -> None:
        """Clean up Redis keys when session ends."""
        self.redis_client.delete(self.queue_key, self.specs_key)
