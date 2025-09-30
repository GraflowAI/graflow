"""Threading-based coordination backend for local parallel execution."""

import concurrent.futures
from typing import TYPE_CHECKING, List, Optional

from graflow.coordination.coordinator import TaskCoordinator

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
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

    def execute_group(self, group_id: str, tasks: List['Executable'], execution_context: 'ExecutionContext') -> None:
        """Execute parallel group using WorkflowEngine in thread pool."""
        self._ensure_executor()
        assert self._executor is not None  # Type checker hint

        print(f"Running parallel group: {group_id}")
        print(f"  Threading tasks: {[task.task_id for task in tasks]}")

        # Check if we have tasks to execute
        if not tasks:
            print(f"  No tasks in group {group_id}")
            return

        def execute_task_with_engine(task: 'Executable') -> tuple[str, bool, str]:
            """Execute single task using WorkflowEngine (thread-safe)."""
            task_id = task.task_id
            try:
                # Create WorkflowEngine per thread for thread safety
                from graflow.core.engine import WorkflowEngine
                engine = WorkflowEngine()

                print(f"  - Executing with WorkflowEngine: {task_id}")
                # Execute single task via unified engine
                engine.execute(execution_context, start_task_id=task_id)

                return task_id, True, "Success"
            except Exception as e:
                error_msg = f"Task {task_id} failed: {e}"
                print(f"    {error_msg}")
                return task_id, False, str(e)

        # Submit all tasks to thread pool with WorkflowEngine execution
        futures = []
        for task in tasks:
            future = self._executor.submit(execute_task_with_engine, task)
            futures.append(future)

        # Wait for all tasks to complete and collect results
        completed_futures = concurrent.futures.as_completed(futures)
        success_count = 0
        failure_count = 0

        for future in completed_futures:
            try:
                task_id, success, message = future.result()
                if success:
                    success_count += 1
                    print(f"  ✓ Task {task_id} completed successfully")
                else:
                    failure_count += 1
                    print(f"  ✗ Task {task_id} failed: {message}")
            except Exception as e:
                failure_count += 1
                print(f"  ✗ Future execution failed: {e}")

        print(f"  Threading group {group_id} completed: {success_count} success, {failure_count} failed")

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
