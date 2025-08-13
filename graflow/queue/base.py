"""Abstract base classes for task queues."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext


class TaskStatus(Enum):
    """Task status management."""
    READY = "ready"        # Ready for execution
    RUNNING = "running"    # Currently executing
    SUCCESS = "success"    # Successfully completed
    ERROR = "error"        # Failed with error


@dataclass
class TaskSpec:
    """Task specification and metadata (Phase 1 implementation)."""
    task_id: str
    execution_context: 'ExecutionContext'
    status: TaskStatus = TaskStatus.READY
    created_at: float = field(default_factory=time.time)
    # Phase 3: Advanced features
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None

    def __lt__(self, other: 'TaskSpec') -> bool:
        """For queue sorting (FIFO: older first)."""
        return self.created_at < other.created_at

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries

    def increment_retry(self, error_message: Optional[str] = None) -> None:
        """Increment retry count and record error."""
        self.retry_count += 1
        self.last_error = error_message
        self.status = TaskStatus.READY  # Reset to ready for retry


class AbstractTaskQueue(ABC):
    """Abstract base class for TaskQueue (TaskSpec support)."""

    def __init__(self, execution_context: 'ExecutionContext'):
        self.execution_context = execution_context
        self._task_specs: Dict[str, TaskSpec] = {}
        # Phase 3: Advanced features
        self.enable_retry: bool = False
        self.enable_metrics: bool = False
        self.metrics: Dict[str, int] = {
            'enqueued': 0,
            'dequeued': 0,
            'retries': 0,
            'failures': 0
        }

    # === Core interface ===
    @abstractmethod
    def enqueue(self, task_spec: TaskSpec) -> bool:
        """Add TaskSpec to queue."""
        pass

    @abstractmethod
    def dequeue(self) -> Optional[TaskSpec]:
        """Get next TaskSpec."""
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    # === Legacy API compatibility ===
    def add_node(self, node_id: str) -> None:
        """Add node ID to queue (ExecutionContext.add_to_queue compatibility)."""
        task_spec = TaskSpec(
            task_id=node_id,
            execution_context=self.execution_context
        )
        self.enqueue(task_spec)

    def get_next_node(self) -> Optional[str]:
        """Get next execution node ID (ExecutionContext.get_next_node compatibility)."""
        task_spec = self.dequeue()
        return task_spec.task_id if task_spec else None

    # === Optional extended interface ===
    def size(self) -> int:
        """Number of waiting nodes."""
        return 0

    def peek_next_node(self) -> Optional[str]:
        """Peek next node without removing."""
        return None

    @abstractmethod
    def to_list(self) -> list[str]:
        """Get list of node IDs in queue order."""
        pass

    def get_task_spec(self, node_id: str) -> Optional[TaskSpec]:
        """Get TaskSpec by node ID."""
        return self._task_specs.get(node_id)

    # === Phase 3: Advanced features ===
    def configure(self, enable_retry: bool = False, enable_metrics: bool = False) -> None:
        """Configure advanced features."""
        self.enable_retry = enable_retry
        self.enable_metrics = enable_metrics

    def handle_task_failure(self, task_spec: TaskSpec, error_message: str) -> bool:
        """Handle task failure with retry logic.

        Returns:
            bool: True if task should be retried, False if it should be marked as failed
        """
        task_spec.status = TaskStatus.ERROR

        if self.enable_metrics:
            self.metrics['failures'] += 1

        if self.enable_retry and task_spec.can_retry():
            task_spec.increment_retry(error_message)
            if self.enable_metrics:
                self.metrics['retries'] += 1
            return True  # Retry

        return False  # Don't retry

    def get_metrics(self) -> Dict[str, int]:
        """Get queue metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset queue metrics."""
        self.metrics = {
            'enqueued': 0,
            'dequeued': 0,
            'retries': 0,
            'failures': 0
        }
