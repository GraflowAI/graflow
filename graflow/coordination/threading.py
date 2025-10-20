"""Threading-based coordination backend for local parallel execution."""

import concurrent.futures
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from graflow.coordination.coordinator import TaskCoordinator

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.handler import TaskHandler
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
                max_workers=self.thread_count,
                thread_name_prefix="ThreadingCoordinator"
            )

    def execute_group(
        self,
        group_id: str,
        tasks: List['Executable'],
        execution_context: 'ExecutionContext',
        handler: 'TaskHandler'
    ) -> None:
        """Execute parallel group using WorkflowEngine in thread pool.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks to execute
            execution_context: Execution context
            handler: TaskHandler instance for group execution
        """
        from graflow.core.handler import TaskResult

        self._ensure_executor()
        assert self._executor is not None  # Type checker hint

        print(f"Running parallel group: {group_id}")
        print(f"  Threading tasks: {[task.task_id for task in tasks]}")

        # Check if we have tasks to execute
        if not tasks:
            print(f"  No tasks in group {group_id}")
            return

        def execute_task_with_engine(task: 'Executable', branch_context: 'ExecutionContext') -> tuple[str, bool, str, float]:
            """Execute single task using WorkflowEngine within a branch context."""
            task_id = task.task_id
            start_time = time.time()
            try:
                from graflow.core.engine import WorkflowEngine

                engine = WorkflowEngine()
                print(f"  - Executing in session '{branch_context.session_id}': {task_id}")
                engine.execute(branch_context, start_task_id=task_id)

                return task_id, True, "Success", time.time() - start_time
            except Exception as e:
                error_msg = f"Task {task_id} failed: {e}"
                print(f"    {error_msg}")
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
                    timestamp=time.time()
                )

                if success:
                    print(f"  ✓ Task {task_id} completed successfully")
                    execution_context.merge_results(branch_context)
                    execution_context.mark_branch_completed(task_id)
                else:
                    print(f"  ✗ Task {task_id} failed: {message}")
            except Exception as e:
                print(f"  ✗ Future execution failed: {e}")
                # Create failure result for unexpected exceptions
                task_id = future_task_map.get(future, "unknown")
                results[task_id] = TaskResult(
                    task_id=task_id,
                    success=False,
                    error_message=str(e),
                    timestamp=time.time()
                )

        print(f"  Threading group {group_id} completed: {len([r for r in results.values() if r.success])} success, {len([r for r in results.values() if not r.success])} failed")

        # Apply handler (can raise ParallelGroupError)
        handler.on_group_finished(group_id, tasks, results, execution_context)

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
