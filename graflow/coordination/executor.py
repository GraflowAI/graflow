"""Parallel execution orchestrator for coordinating task groups."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from graflow.coordination.coordinator import CoordinationBackend, TaskCoordinator
from graflow.coordination.redis_coordinator import RedisCoordinator
from graflow.coordination.threading_coordinator import ThreadingCoordinator
from graflow.queue.distributed import DistributedTaskQueue

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.handlers.group_policy import GroupExecutionPolicy
    from graflow.core.task import Executable

logger = logging.getLogger(__name__)


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
        backend: CoordinationBackend, config: Dict[str, Any], exec_context: ExecutionContext
    ) -> TaskCoordinator:
        """Create appropriate coordinator based on backend."""
        if backend == CoordinationBackend.REDIS:
            # Create Redis client from connection parameters
            from graflow.utils.redis_utils import create_redis_client

            # Ensure decode_responses=True for DistributedTaskQueue
            # (only set if not already specified)
            if "decode_responses" not in config:
                config = {**config, "decode_responses": True}

            try:
                redis_client = create_redis_client(config)
                key_prefix = config.get("key_prefix", "graflow")
                task_queue = DistributedTaskQueue(redis_client=redis_client, key_prefix=key_prefix)
            except ImportError as e:
                raise ImportError("Redis backend requires 'redis' package") from e

            return RedisCoordinator(task_queue)

        if backend == CoordinationBackend.THREADING:
            thread_count = config.get("thread_count")
            return ThreadingCoordinator(thread_count)

        if backend == CoordinationBackend.DIRECT:
            raise ValueError("DIRECT backend should be handled separately")

        raise ValueError(f"Unsupported backend: {backend}")

    @staticmethod
    def execute_parallel_group(
        group_id: str,
        tasks: List[Executable],
        exec_context: ExecutionContext,
        *,
        backend: Optional[Union[str, CoordinationBackend]] = None,
        backend_config: Optional[Dict[str, Any]] = None,
        policy: Union[str, GroupExecutionPolicy] = "strict",
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
        resolved_backend = GroupExecutor._resolve_backend(backend)

        # Merge context config with backend config
        # backend_config takes precedence over context config
        context_config = getattr(exec_context, "config", {})
        config = {**context_config, **(backend_config or {})}

        from graflow.core.handlers.group_policy import resolve_group_policy

        policy_instance = resolve_group_policy(policy)

        if resolved_backend == CoordinationBackend.DIRECT:
            return GroupExecutor.direct_execute(group_id, tasks, exec_context, policy_instance)

        coordinator = GroupExecutor._create_coordinator(resolved_backend, config, exec_context)
        coordinator.execute_group(group_id, tasks, exec_context, policy_instance)

    @staticmethod
    def direct_execute(
        group_id: str,
        tasks: List[Executable],
        execution_context: ExecutionContext,
        policy_instance: GroupExecutionPolicy,
    ) -> None:
        """Execute tasks using unified WorkflowEngine for consistency."""
        from graflow.core.handler import TaskResult

        task_ids = [task.task_id for task in tasks]
        logger.info(
            "Running parallel group: %s with %d tasks",
            group_id,
            len(tasks),
            extra={"group_id": group_id, "task_ids": task_ids},
        )
        logger.debug("Direct tasks: %s", task_ids)

        # Use unified WorkflowEngine for each task
        from graflow.core.engine import WorkflowEngine

        engine = WorkflowEngine()
        results: Dict[str, TaskResult] = {}

        for task in tasks:
            logger.debug("Executing task directly: %s", task.task_id, extra={"group_id": group_id})
            success = True
            error_message = None
            start_time = time.time()
            try:
                # Execute single task via unified engine
                engine.execute(execution_context, start_task_id=task.task_id)
            except Exception as e:
                logger.error(
                    "Task failed in parallel group: %s",
                    task.task_id,
                    exc_info=True,
                    extra={"group_id": group_id, "error": str(e)},
                )
                success = False
                error_message = str(e)

            results[task.task_id] = TaskResult(
                task_id=task.task_id,
                success=success,
                error_message=error_message,
                duration=time.time() - start_time,
                timestamp=time.time(),
            )

        logger.info(
            "Direct parallel group completed: %s",
            group_id,
            extra={
                "group_id": group_id,
                "task_count": len(tasks),
                "success_count": sum(1 for r in results.values() if r.success),
            },
        )

        # Use GroupExecutionPolicy directly instead of handler
        policy_instance.on_group_finished(group_id, tasks, results, execution_context)
