"""Threading-based coordination backend for local parallel execution."""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from graflow.coordination.coordinator import TaskCoordinator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.handlers.group_policy import GroupExecutionPolicy
    from graflow.core.task import Executable


class ThreadingCoordinator(TaskCoordinator):
    """Threading-based task coordination for local parallel execution."""

    def __init__(self, thread_count: Optional[int] = None):
        """Initialize threading coordinator."""
        import multiprocessing as mp

        self.thread_count = thread_count or mp.cpu_count()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def _ensure_executor(self) -> None:
        """Ensure thread pool executor is initialized."""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.thread_count, thread_name_prefix="ThreadingCoordinator"
            )

    def execute_group(
        self, group_id: str, tasks: List[Executable], execution_context: ExecutionContext, policy: GroupExecutionPolicy
    ) -> None:
        """Execute parallel group using WorkflowEngine in thread pool.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            execution_context: Execution context
            policy: GroupExecutionPolicy instance for result evaluation
        """
        from graflow.core.handler import TaskResult

        self._ensure_executor()
        assert self._executor is not None  # Type checker hint

        logger.info("Running parallel group: %s", group_id)
        logger.debug("Threading tasks: %s", [task.task_id for task in tasks])

        # Check if we have tasks to execute
        if not tasks:
            logger.info("No tasks in group %s", group_id)
            return

        def execute_task_with_engine(
            task: Executable, branch_context: ExecutionContext
        ) -> tuple[str, bool, str, float]:
            """Execute single task using WorkflowEngine within a branch context.

            Note: The tracer is propagated via branch_context, which inherits the parent tracer.
            Each parallel task will create its own span within the parent trace.
            """
            task_id = task.task_id
            start_time = time.time()
            try:
                from graflow.core.engine import WorkflowEngine

                # Create engine with the branch context (which has the tracer)
                engine = WorkflowEngine()
                logger.debug(
                    "Executing task %s in session '%s' (tracer: %s)",
                    task_id,
                    branch_context.session_id,
                    "enabled" if branch_context.tracer else "disabled",
                )

                # Execute task - the engine will use branch_context.tracer for tracing
                engine.execute(branch_context, start_task_id=task_id)

                return task_id, True, "Success", time.time() - start_time
            except Exception as e:
                error_msg = f"Task {task_id} failed: {e}"
                logger.debug(error_msg)
                return task_id, False, str(e), time.time() - start_time

        # Submit all tasks to thread pool with isolated branch contexts
        futures = []
        future_context_map: dict[concurrent.futures.Future, ExecutionContext] = {}
        future_task_map: Dict[concurrent.futures.Future, str] = {}

        for task in tasks:
            branch_context = execution_context.create_branch_context(branch_id=task.task_id)

            future = self._executor.submit(execute_task_with_engine, task, branch_context)
            futures.append(future)
            future_context_map[future] = branch_context
            future_task_map[future] = task.task_id

        # Wait for all tasks to complete and collect results
        completed_futures = concurrent.futures.as_completed(futures)
        results: Dict[str, TaskResult] = {}

        for future in completed_futures:
            branch_context = future_context_map[future]
            try:
                task_id, success, message, duration = future.result()

                # Create TaskResult
                results[task_id] = TaskResult(
                    task_id=task_id,
                    success=success,
                    error_message=message if not success else None,
                    duration=duration,
                    timestamp=time.time(),
                )

                if success:
                    logger.info("Task %s completed successfully", task_id)
                    execution_context.merge_results(branch_context)
                    execution_context.mark_branch_completed(task_id)
                else:
                    logger.warning("Task %s failed: %s", task_id, message)
            except Exception as e:
                logger.error("Future execution failed: %s", e)
                # Create failure result for unexpected exceptions
                task_id = future_task_map.get(future, "unknown")
                results[task_id] = TaskResult(
                    task_id=task_id, success=False, error_message=str(e), timestamp=time.time()
                )

        success_count = len([r for r in results.values() if r.success])
        failure_count = len(results) - success_count
        logger.info(
            "Threading group %s completed: %d success, %d failed",
            group_id,
            success_count,
            failure_count,
        )

        # Apply policy directly (can raise ParallelGroupError)
        policy.on_group_finished(group_id, tasks, results, execution_context)

    def shutdown(self) -> None:
        """Shutdown the coordinator and cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self) -> None:
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except Exception:
            pass
