"""Task queue factory for creating different backend implementations."""

from enum import Enum
from typing import Any, Dict

from .base import AbstractTaskQueue
from .memory import InMemoryTaskQueue


class QueueBackend(Enum):
    """Available queue backends."""
    IN_MEMORY = "in_memory"           # in-memory FIFO queue (Phase 1)
    REDIS = "redis"                   # Redis distributed queue (Phase 2)


class TaskQueueFactory:
    """Factory class for TaskQueue."""

    @staticmethod
    def create(backend: QueueBackend, execution_context, **kwargs) -> AbstractTaskQueue:
        """Create TaskQueue with specified backend."""

        if backend == QueueBackend.IN_MEMORY:
            start_node = kwargs.get('start_node')
            return InMemoryTaskQueue(execution_context, start_node)

        elif backend == QueueBackend.REDIS:
            from .redis import RedisTaskQueue  # noqa: PLC0415
            redis_config = kwargs.get('redis_config', {})
            redis_client = kwargs.get('redis_client')
            key_prefix = kwargs.get('key_prefix') or redis_config.get('key_prefix', 'graflow')
            return RedisTaskQueue(execution_context, redis_client, key_prefix)

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    @staticmethod
    def create_from_config(execution_context, config: Dict[str, Any]) -> AbstractTaskQueue:
        """Create TaskQueue from configuration dictionary."""
        backend_name = config.get('backend', 'in_memory')
        backend = QueueBackend(backend_name)

        # Extract backend-specific config
        backend_config = config.get('config', {})

        return TaskQueueFactory.create(backend, execution_context, **backend_config)
