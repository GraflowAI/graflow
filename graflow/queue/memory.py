"""In-memory task queue implementation."""

from collections import deque
from typing import Optional

from graflow.queue.base import AbstractTaskQueue, TaskSpec, TaskStatus


class InMemoryTaskQueue(AbstractTaskQueue):
    """In-memory task queue with TaskSpec support (Phase 1 implementation)."""

    def __init__(self, execution_context, start_node: Optional[str] = None):
        super().__init__(execution_context)
        self._queue = deque()
        if start_node:
            task_spec = TaskSpec(
                node_id=start_node,
                execution_context=execution_context
            )
            self._queue.append(task_spec)
            self._task_specs[start_node] = task_spec

    def enqueue(self, task_spec: TaskSpec) -> bool:
        """Add TaskSpec to queue (FIFO)."""
        self._queue.append(task_spec)
        self._task_specs[task_spec.node_id] = task_spec

        # Phase 3: Metrics
        if self.enable_metrics:
            self.metrics['enqueued'] += 1

        return True

    def dequeue(self) -> Optional[TaskSpec]:
        """Get next TaskSpec."""
        if self._queue:
            task_spec = self._queue.popleft()
            task_spec.status = TaskStatus.RUNNING

            # Phase 3: Metrics
            if self.enable_metrics:
                self.metrics['dequeued'] += 1

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
        return self._queue[0].node_id if self._queue else None

    def to_list(self) -> list[str]:
        """Get list of node IDs in queue order."""
        return [task_spec.node_id for task_spec in self._queue]

    # === Phase 3: Advanced features ===
    def retry_failed_task(self, task_spec: TaskSpec) -> bool:
        """Re-enqueue a failed task for retry."""
        if self.enable_retry and task_spec.can_retry():
            # Reset status and re-enqueue
            task_spec.status = TaskStatus.READY
            self.enqueue(task_spec)
            return True
        return False
