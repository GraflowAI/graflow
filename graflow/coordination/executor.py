"""Parallel execution orchestrator for coordinating task groups."""

from typing import Any, Dict, List, Optional

import redis

from graflow.coordination.coordinator import CoordinationBackend, TaskCoordinator
from graflow.coordination.multiprocessing import MultiprocessingCoordinator
from graflow.coordination.redis import RedisCoordinator
from graflow.coordination.task_spec import TaskSpec


class GroupExecutor:
    """Unified executor for parallel task groups supporting multiple backends."""

    def __init__(self, backend: CoordinationBackend = CoordinationBackend.MULTIPROCESSING,
                 backend_config: Optional[Dict[str, Any]] = None):
        self.backend: CoordinationBackend = backend
        self.coordinator: Optional[TaskCoordinator] = self._create_coordinator(backend, backend_config or {})
        self.execution_context: Optional[Any] = None

    def _create_coordinator(self, backend: CoordinationBackend, config: Dict[str, Any]) -> Optional[TaskCoordinator]:
        """Create appropriate coordinator based on backend."""
        if backend == CoordinationBackend.DIRECT:
            return None  # No coordinator needed for direct execution

        elif backend == CoordinationBackend.REDIS:
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
        if self.backend == CoordinationBackend.DIRECT:
            return self.direct_execute(group_id, tasks)
        else:
            assert self.coordinator is not None, "Coordinator must be initialized for parallel execution"
            return self.coordinator.execute_group(group_id, tasks)

    def direct_execute(self, group_id: str, tasks: List[TaskSpec]) -> None:
        """Directly execute tasks without coordination."""
        print(f"Running parallel group: {group_id}")
        print(f"  Direct tasks: {[task.task_id for task in tasks]}")

        for task in tasks:
            print(f"  - Executing directly: {task.task_id}")
            task.func(*task.args, **task.kwargs)

        print(f"  Direct group {group_id} completed")
