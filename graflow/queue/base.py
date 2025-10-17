"""Abstract base classes for task queues."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from graflow.core.context import ExecutionContext
    from graflow.core.task import Executable

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status management."""
    READY = "ready"        # Ready for execution
    RUNNING = "running"    # Currently executing
    SUCCESS = "success"    # Successfully completed
    ERROR = "error"        # Failed with error


@dataclass
class TaskSpec:
    """Task specification and metadata with function serialization support."""
    executable: 'Executable'
    execution_context: 'ExecutionContext'
    strategy: str = "reference"
    status: TaskStatus = TaskStatus.READY
    created_at: float = field(default_factory=time.time)
    # Phase 3: Advanced features
    retry_count: int = 0
    max_retries: int = 3
    last_error: Optional[str] = None
    # Phase 2: Barrier synchronization support
    group_id: Optional[str] = None

    @property
    def task_id(self) -> str:
        """Get task_id from executable."""
        return self.executable.task_id

    @property
    def function_data(self) -> Dict[str, Any]:
        """Get function_data by serializing executable's function."""
        return self.execution_context.function_manager.serialize_task(self.executable, self.strategy)

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

    def get_function(self) -> Optional['Executable']:
        """Get function for this task by deserializing stored data.

        Returns:
            Function object or None if no function data available

        Raises:
            TaskResolutionError: If task cannot be resolved
        """
        try:
            function_data = self.function_data
            return self.execution_context.function_manager.resolve_task(function_data)
        except ValueError:
            return None
        except Exception:
            return None

class TaskQueue(ABC):
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

    def get_next_task(self) -> Optional[str]:
        """Get next execution node ID (ExecutionContext.get_next_task compatibility)."""
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
        """Get list of task IDs in queue order."""
        pass

    def get_task_spec(self, task_id: str) -> Optional[TaskSpec]:
        """Get TaskSpec by task ID."""
        return self._task_specs.get(task_id)

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

    def notify_task_completion(self, task_id: str, success: bool,
                             group_id: Optional[str] = None) -> None:
        """Notify task completion for barrier synchronization.

        Default implementation does nothing. Subclasses can override
        to implement barrier synchronization logic.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            group_id: Group ID for barrier synchronization
        """
        # Default implementation does nothing
        logger.debug(f"Task {task_id} completed with success={success} in group {group_id}")
        pass
