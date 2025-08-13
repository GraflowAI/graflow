"""Threading-based coordination backend for local parallel execution."""

import concurrent.futures
from typing import List, Optional

from graflow.coordination.coordinator import TaskCoordinator
from graflow.coordination.task_spec import TaskSpec


class MultiprocessingCoordinator(TaskCoordinator):
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
                thread_name_prefix="MultiprocessingCoordinator"
            )

    def execute_group(self, group_id: str, tasks: List[TaskSpec]) -> None:
        """Execute parallel group using threading."""
        self._ensure_executor()
        assert self._executor is not None  # Type checker hint

        print(f"Running parallel group: {group_id}")
        print(f"  Threading tasks: {[task.task_id for task in tasks]}")

        # Submit all tasks to thread pool
        futures = []
        for task in tasks:
            print(f"  - Executing with threading: {task.task_id}")
            future = self._executor.submit(task.func, *task.args, **task.kwargs)
            futures.append(future)

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
        print(f"  Threading group {group_id} completed")

    def shutdown(self) -> None:
        """Shutdown the coordinator and cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except Exception:
            pass
