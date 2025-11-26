"""Parallel execution orchestrator for coordinating task groups."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from graflow.coordination.coordinator import CoordinationBackend, TaskCoordinator
from graflow.coordination.redis import RedisCoordinator
from graflow.coordination.threading import ThreadingCoordinator
from graflow.queue.redis import DistributedTaskQueue

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.handler import TaskHandler
    from graflow.core.handlers.group_policy import GroupExecutionPolicy
    from graflow.core.task import Executable


class GroupExecutor:
    """Unified executor for parallel task groups supporting multiple backends.

    This executor is stateless. It creates appropriate coordinators per execution
    request based on the specified backend and configuration.
    """

    DEFAULT_BACKEND: CoordinationBackend = CoordinationBackend.THREADING

    @staticmethod
    def _resolve_backend(backend: Optional[Union[str, CoordinationBackend]]) -> CoordinationBackend:
        if backend is None:
            return GroupExecutor.DEFAULT_BACKEND
        if isinstance(backend, CoordinationBackend):
            return backend
        if isinstance(backend, str):
            normalized = backend.lower()
            for candidate in CoordinationBackend:
                if candidate.value == normalized or candidate.name.lower() == normalized:
                    return candidate
        raise ValueError(f"Unsupported coordination backend: {backend}")

    @staticmethod
    def _create_coordinator(
        backend: CoordinationBackend,
        config: Dict[str, Any],
        exec_context: 'ExecutionContext'
    ) -> TaskCoordinator:
        """Create appropriate coordinator based on backend."""
        if backend == CoordinationBackend.REDIS:
            redis_kwargs: Dict[str, Any] = {}
            if "redis_client" in config:
                redis_kwargs["redis_client"] = config["redis_client"]
            else:
                redis_kwargs["host"] = config.get("host", "localhost")
                redis_kwargs["port"] = config.get("port", 6379)
                redis_kwargs["db"] = config.get("db", 0)
            redis_kwargs["key_prefix"] = config.get("key_prefix", "graflow")

            try:
                task_queue = DistributedTaskQueue(exec_context, **redis_kwargs)
            except ImportError as e:
                raise ImportError("Redis backend requires 'redis' package") from e

            return RedisCoordinator(task_queue)

        if backend == CoordinationBackend.THREADING:
            thread_count = config.get("thread_count")
            return ThreadingCoordinator(thread_count)

        if backend == CoordinationBackend.DIRECT:
            raise ValueError("DIRECT backend should be handled separately")

        raise ValueError(f"Unsupported backend: {backend}")

    def execute_parallel_group(
        self,
        group_id: str,
        tasks: List['Executable'],
        exec_context: 'ExecutionContext',
        *,
        backend: Optional[Union[str, CoordinationBackend]] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        policy: Union[str, 'GroupExecutionPolicy'] = "strict",
    ) -> None:
        """Execute parallel group with a configurable group policy.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            exec_context: Execution context
            backend: Coordination backend (name or CoordinationBackend)
            backend_config: Backend-specific configuration
            policy: Group execution policy (name or instance)
        """
        from graflow.core.handlers.direct import DirectTaskHandler
        from graflow.core.handlers.group_policy import resolve_group_policy

        policy_instance = resolve_group_policy(policy)

        handler = DirectTaskHandler()
        handler.set_group_policy(policy_instance)

        resolved_backend = self._resolve_backend(backend)

        # Merge context config with backend config
        # backend_config takes precedence over context config
        context_config = getattr(exec_context, 'config', {})
        config = {**context_config, **(backend_config or {})}

        if resolved_backend == CoordinationBackend.DIRECT:
            return self.direct_execute(group_id, tasks, exec_context, handler)

        coordinator = self._create_coordinator(resolved_backend, config, exec_context)
        coordinator.execute_group(group_id, tasks, exec_context, handler)

    def direct_execute(
        self,
        group_id: str,
        tasks: List['Executable'],
        execution_context: 'ExecutionContext',
        handler: 'TaskHandler'
    ) -> None:
        """Execute tasks using unified WorkflowEngine for consistency."""
        import time

        from graflow.core.handler import TaskResult

        print(f"Running parallel group: {group_id}")
        print(f"  Direct tasks: {[task.task_id for task in tasks]}")

        # Use unified WorkflowEngine for each task
        from graflow.core.engine import WorkflowEngine

        engine = WorkflowEngine()
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

            results[task.task_id] = TaskResult(
                task_id=task.task_id,
                success=success,
                error_message=error_message,
                duration=time.time() - start_time,
                timestamp=time.time()
            )

        print(f"  Direct group {group_id} completed")

        handler.on_group_finished(group_id, tasks, results, execution_context)
