"""Parallel execution orchestrator for coordinating task groups."""

from typing import Any, Dict, List, Optional

import redis

from graflow.coordination.coordinator import CoordinationBackend, TaskCoordinator
from graflow.coordination.multiprocessing import MultiprocessingCoordinator
from graflow.coordination.redis import RedisCoordinator


class TaskSpec:
    """Task specification for parallel execution."""

    def __init__(self, task_id: str, func, args: tuple = (), kwargs: Optional[dict] = None):
        self.task_id = task_id
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}


class GroupExecutor:
    """Unified executor for parallel task groups supporting multiple backends."""

    def __init__(self, backend: CoordinationBackend = CoordinationBackend.MULTIPROCESSING,
                 backend_config: Optional[Dict[str, Any]] = None):
        self.backend = backend
        self.coordinator = self._create_coordinator(backend, backend_config or {})
        self.execution_context: Optional[Any] = None

    def _create_coordinator(self, backend: CoordinationBackend, config: Dict[str, Any]) -> TaskCoordinator:
        """Create appropriate coordinator based on backend."""
        if backend == CoordinationBackend.REDIS:
            try:
                redis_client = config.get("redis_client")
                if redis_client is None:
                    redis_client = redis.Redis(
                        host=config.get("host", "localhost"),
                        port=config.get("port", 6379),
                        db=config.get("db", 0)
                    )
                return RedisCoordinator(redis_client)
            except ImportError as e:
                raise ImportError("Redis backend requires 'redis' package") from e

        elif backend == CoordinationBackend.MULTIPROCESSING:
            process_count = config.get("process_count")
            return MultiprocessingCoordinator(process_count)

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def set_execution_context(self, context: Any) -> None:
        """Set execution context for result tracking."""
        self.execution_context = context

    def execute_parallel_group(self, group_id: str, tasks: List[TaskSpec]) -> None:
        """Execute parallel group with barrier synchronization."""
        # Create barrier
        barrier_id = self.coordinator.create_barrier(group_id, len(tasks))

        try:
            # Dispatch all tasks
            for task in tasks:
                self.coordinator.dispatch_task(task, group_id)

            # Wait for all tasks to complete
            if not self.coordinator.wait_barrier(barrier_id):
                raise TimeoutError(f"Barrier wait timeout for group {group_id}")

            print(f"Parallel group {group_id} completed successfully")

        finally:
            # Clean up barrier
            self.coordinator.cleanup_barrier(barrier_id)
