"""Parallel execution orchestrator for coordinating task groups."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from graflow.coordination.coordinator import CoordinationBackend, TaskCoordinator
from graflow.coordination.redis import RedisCoordinator
from graflow.coordination.threading import ThreadingCoordinator
from graflow.queue.base import TaskQueue
from graflow.queue.redis import RedisTaskQueue

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.handler import TaskHandler
    from graflow.core.task import Executable


class GroupExecutor:
    """Unified executor for parallel task groups supporting multiple backends."""

    def __init__(self, backend: CoordinationBackend = CoordinationBackend.THREADING,
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

        elif backend == CoordinationBackend.THREADING:
            thread_count = config.get("thread_count")
            return ThreadingCoordinator(thread_count)

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def execute_parallel_group(
        self,
        group_id: str,
        tasks: List['Executable'],
        exec_context: 'ExecutionContext',
        handler: Optional['TaskHandler'] = None
    ) -> None:
        """Execute parallel group with handler.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            exec_context: Execution context
            handler: TaskHandler instance (None = use default DirectTaskHandler)
        """
        # Use handler instance directly or create default
        if handler is None:
            from graflow.core.handlers.direct import DirectTaskHandler
            handler = DirectTaskHandler()

        # Execute with appropriate backend
        if self.backend == CoordinationBackend.DIRECT:
            return self.direct_execute(group_id, tasks, exec_context, handler)
        else:
            coordinator = self._create_coordinator(self.backend, self.backend_config, exec_context)
            return coordinator.execute_group(group_id, tasks, exec_context, handler)

    def direct_execute(
        self,
        group_id: str,
        tasks: List['Executable'],
        execution_context: 'ExecutionContext',
        handler: 'TaskHandler'
    ) -> None:
        """Execute tasks using unified WorkflowEngine for consistency.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            execution_context: Execution context
            handler: TaskHandler instance for group policy
        """
        import time

        from graflow.core.handler import TaskResult

        print(f"Running parallel group: {group_id}")
        print(f"  Direct tasks: {[task.task_id for task in tasks]}")

        # Use unified WorkflowEngine for each task
        from graflow.core.engine import WorkflowEngine
        engine = WorkflowEngine()

        # Collect results
        results: Dict[str, TaskResult] = {}

        for task in tasks:
            print(f"  - Executing directly: {task.task_id}")
            success = True
            error_message = None
            start_time = time.time()
            try:
                # Execute single task via unified engine
                engine.execute(execution_context, start_task_id=task.task_id)
            except Exception as e:
                print(f"    Task {task.task_id} failed: {e}")
                success = False
                error_message = str(e)
                # Continue with other tasks in the group

            # Record result
            results[task.task_id] = TaskResult(
                task_id=task.task_id,
                success=success,
                error_message=error_message,
                duration=time.time() - start_time,
                timestamp=time.time()
            )

        print(f"  Direct group {group_id} completed")

        # Apply handler (can raise ParallelGroupError)
        handler.on_group_finished(group_id, tasks, results, execution_context)
