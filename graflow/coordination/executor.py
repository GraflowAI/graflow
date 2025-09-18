"""Parallel execution orchestrator for coordinating task groups."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graflow.coordination.coordinator import CoordinationBackend, TaskCoordinator
from graflow.coordination.multiprocessing import MultiprocessingCoordinator
from graflow.coordination.redis import RedisCoordinator
from graflow.queue.base import TaskQueue
from graflow.queue.redis import RedisTaskQueue

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.task import Executable


class GroupExecutor:
    """Unified executor for parallel task groups supporting multiple backends."""

    def __init__(self, backend: CoordinationBackend = CoordinationBackend.MULTIPROCESSING,
                 backend_config: Optional[Dict[str, Any]] = None):
        self.backend: CoordinationBackend = backend
        self.backend_config: Dict[str, Any] = backend_config or {}

    def _create_coordinator(self, backend: CoordinationBackend, config: Dict[str, Any], exec_context: 'ExecutionContext') -> TaskCoordinator:
        """Create appropriate coordinator based on backend."""
        if backend == CoordinationBackend.REDIS:
            try:
                task_queue: TaskQueue = exec_context.queue
                if not isinstance(task_queue, RedisTaskQueue):
                    raise ValueError(f"Execution context must provide a valid RedisTaskQueue for Redis backend: {type(task_queue)}")
                return RedisCoordinator(task_queue)
            except ImportError as e:
                raise ImportError("Redis backend requires 'redis' package") from e

        elif backend == CoordinationBackend.MULTIPROCESSING:
            process_count = config.get("process_count")
            return MultiprocessingCoordinator(process_count)

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def execute_parallel_group(self, group_id: str, tasks: List['Executable'], exec_context: 'ExecutionContext') -> None:
        """Execute parallel group with barrier synchronization."""
        if self.backend == CoordinationBackend.DIRECT:
            return self.direct_execute(group_id, tasks, exec_context)
        else:
            coordinator = self._create_coordinator(self.backend, self.backend_config, exec_context)
            return coordinator.execute_group(group_id, tasks)

    def direct_execute(self, group_id: str, tasks: List['Executable'], execution_context: 'ExecutionContext') -> None:
        """Directly execute tasks without coordination."""
        print(f"Running parallel group: {group_id}")
        print(f"  Direct tasks: {[task.task_id for task in tasks]}")

        for task in tasks:
            print(f"  - Executing directly: {task.task_id}")

            # Execute task with proper context management (same as engine)
            try:
                with execution_context.executing_task(task) as _ctx:
                    result = task.run()
                    execution_context.set_result(task.task_id, result)
                    # No context.increment_step() here - individual tasks don't increment
            except Exception as e:
                execution_context.set_result(task.task_id, e)
                print(f"    Task {task.task_id} failed: {e}")
                # Continue with other tasks in the group

        # Only increment step once for the entire parallel group
        execution_context.increment_step()
        print(f"  Direct group {group_id} completed")
