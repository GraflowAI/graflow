"""In-memory task queue implementation."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional

from graflow.queue.base import TaskQueue, TaskSpec, TaskStatus

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext


class LocalTaskQueue(TaskQueue):
    """In-memory task queue with TaskSpec support."""

    def __init__(self, execution_context: ExecutionContext, start_node: Optional[str] = None):
        super().__init__()
        self._queue: deque[TaskSpec] = deque()
        if start_node:
            # Get task from graph instead of creating a new Task object
            # This ensures single source of truth and preserves task metadata
            try:
                start_task = execution_context.graph.get_node(start_node)

                if start_task:
                    task_spec = TaskSpec(executable=start_task, execution_context=execution_context)
                    self._queue.append(task_spec)
                    self._task_specs[start_node] = task_spec
                else:
                    available_nodes = list(execution_context.graph.nodes.keys())
                    self._logger.warning(
                        f"Start node '{start_node}' not found in graph. Available nodes: {available_nodes}"
                    )
            except (AttributeError, KeyError) as e:
                # Graph might not be fully deserialized yet during unpickling
                # This is okay - the queue will be populated later when tasks are added
                self._logger.debug(
                    f"Could not access graph during LocalTaskQueue initialization: {e}. "
                    f"This is expected during deserialization."
                )

    def enqueue(self, task_spec: TaskSpec) -> bool:
        """Add TaskSpec to queue (FIFO)."""
        self._queue.append(task_spec)
        self._task_specs[task_spec.task_id] = task_spec

        if self.enable_metrics:
            self.metrics["enqueued"] += 1

        return True

    def dequeue(self) -> Optional[TaskSpec]:
        """Get next TaskSpec."""
        if self._queue:
            task_spec = self._queue.popleft()
            task_spec.status = TaskStatus.RUNNING

            if self.enable_metrics:
                self.metrics["dequeued"] += 1

            return task_spec
        return None

    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return len(self._queue) == 0

    def cleanup(self) -> None:
        """Cleanup resources."""
        self._queue.clear()

    def size(self) -> int:
        """Number of waiting TaskSpecs."""
        return len(self._queue)

    def peek_next_node(self) -> Optional[str]:
        """Peek next node without removing."""
        return self._queue[0].task_id if self._queue else None

    def to_list(self) -> list[str]:
        """Get list of node IDs in queue order."""
        return [task_spec.task_id for task_spec in self._queue]

    def get_pending_task_specs(self) -> list[TaskSpec]:
        """Return pending TaskSpec objects (shallow copy)."""
        return list(self._queue)

    def retry_failed_task(self, task_spec: TaskSpec) -> bool:
        """Re-enqueue a failed task for retry."""
        if self.enable_retry and task_spec.can_retry():
            # Reset status and re-enqueue
            task_spec.status = TaskStatus.READY
            self.enqueue(task_spec)
            return True
        return False
