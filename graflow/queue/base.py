"""Abstract base classes for task queues."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.task import Executable


class TaskStatus(Enum):
    """Task status management."""

    READY = "ready"  # Ready for execution
    RUNNING = "running"  # Currently executing
    SUCCESS = "success"  # Successfully completed
    ERROR = "error"  # Failed with error


@dataclass
class TaskSpec:
    """Task specification and metadata for distributed task execution.

    TaskSpec stores task execution metadata and maintains a reference to the
    executable task. The actual task is stored in the Graph (via GraphStore)
    and retrieved directly without additional serialization.
    """

    executable: Executable
    execution_context: ExecutionContext
    status: TaskStatus = TaskStatus.READY
    created_at: float = field(default_factory=time.time)
    # Phase 3: Advanced features
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    # Phase 2: Barrier synchronization support
    group_id: Optional[str] = None
    # Trace context for distributed tracing
    trace_id: Optional[str] = None  # Trace ID (= session_id, W3C compliant 32-digit hex)
    parent_span_id: Optional[str] = None  # Parent span ID (task that queued this task)

    @property
    def task_id(self) -> str:
        """Get task_id from executable."""
        return self.executable.task_id

    def __lt__(self, other: TaskSpec) -> bool:
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

    def get_task(self) -> Optional[Executable]:
        """Get task executable from TaskSpec.

        The task is already resolved from the Graph, so this simply
        returns the executable directly without additional serialization.

        Returns:
            Executable object (already resolved from Graph)
        """
        return self.executable


class TaskQueue(ABC):
    """Abstract base class for all task queues."""

    def __init__(self):
        """Initialize TaskQueue."""
        self._task_specs: Dict[str, TaskSpec] = {}
        # Advanced features
        self.enable_retry: bool = False
        self.enable_metrics: bool = False
        self.metrics: Dict[str, int] = {"enqueued": 0, "dequeued": 0, "retries": 0, "failures": 0}

    @property
    def _logger(self):
        """Get logger instance (lazy initialization to avoid module-level import)."""
        if not hasattr(self, "_logger_instance"):
            import logging

            self._logger_instance = logging.getLogger(__name__)
        return self._logger_instance

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

    def get_next_task(self) -> Optional[str]:
        """Get next execution node ID (ExecutionContext.get_next_task compatibility)."""
        task_spec = self.dequeue()
        return task_spec.task_id if task_spec else None

    # === Common optional interface ===
    def size(self) -> int:
        """Number of waiting nodes."""
        return 0

    def get_task_spec(self, task_id: str) -> Optional[TaskSpec]:
        """Get TaskSpec by task ID."""
        return self._task_specs.get(task_id)

    # === Advanced features ===
    def configure(self, enable_retry: bool = False, enable_metrics: bool = False) -> None:
        """Configure advanced features."""
        self.enable_retry = enable_retry
        self.enable_metrics = enable_metrics

    def handle_task_failure(self, task_spec: TaskSpec, error_message: str) -> bool:
        """Handle task failure with retry logic."""
        task_spec.status = TaskStatus.ERROR

        if self.enable_metrics:
            self.metrics["failures"] += 1

        if self.enable_retry and task_spec.can_retry():
            task_spec.increment_retry(error_message)
            if self.enable_metrics:
                self.metrics["retries"] += 1
            return True  # Retry

        return False  # Don't retry

    def get_metrics(self) -> Dict[str, int]:
        """Get queue metrics."""
        return self.metrics.copy()

    def reset_metrics(self) -> None:
        """Reset queue metrics."""
        self.metrics = {"enqueued": 0, "dequeued": 0, "retries": 0, "failures": 0}

    def notify_task_completion(
        self, task_id: str, success: bool, group_id: Optional[str] = None, error_message: Optional[str] = None
    ) -> None:
        """Notify task completion for barrier synchronization.

        Default implementation does nothing. Subclasses can override
        to implement barrier synchronization logic.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            group_id: Group ID for barrier synchronization
            error_message: Error message if task failed
        """
        # Default implementation does nothing
        if success:
            self._logger.debug(f"Task {task_id} completed successfully in group {group_id}")
        else:
            self._logger.debug(f"Task {task_id} failed in group {group_id}: {error_message}")
