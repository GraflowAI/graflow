"""Task execution handler base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from graflow.core.context import ExecutionContext
from graflow.core.task import Executable


@dataclass
class TaskResult:
    """Result of a task execution.

    Note: Task result values are stored in ExecutionContext and can be
    accessed via context.get_result(task_id) or channels if needed.
    This dataclass focuses on execution status, not result values.
    """

    task_id: str
    success: bool
    error: Optional[Exception] = None
    error_message: Optional[str] = None
    duration: float = 0.0
    timestamp: float = 0.0


class TaskHandler(ABC):
    """Base class for task execution handlers.

    TaskHandler is responsible for individual task execution strategy.
    For group result evaluation, use GroupExecutionPolicy directly.

    Design:
    - execute_task() is abstract - all handlers must provide execution strategy
    - get_name() has default implementation (class name)

    Responsibility Separation:
    - TaskHandler: How individual tasks execute (via @task(handler="..."))
    - GroupExecutionPolicy: How parallel groups succeed/fail (via .with_execution(policy=...))
    """

    def get_name(self) -> str:
        """Get handler name for registration.

        Returns:
            Handler name (used for registration and lookup)

        Note:
            This method has a default implementation for backward compatibility.
            Override to provide a custom name.

        Examples:
            >>> # Execution handler: inherits from TaskHandler
            >>> class DockerTaskHandler(TaskHandler):
            ...     def get_name(self):
            ...         return "docker"
            ...     def execute_task(self, task, context):
            ...         # ... implementation
            >>>
            >>> # Policy handler: can reuse built-in names
            >>> from graflow.core.handlers import AtLeastNGroupPolicy
            >>> handler = AtLeastNGroupPolicy(min_success=2)
            >>> handler.get_name()
            'at_least_2'
            >>> # Default get_name() implementation (no override needed)
            >>> class MyHandler(TaskHandler):
            ...     # get_name() inherited -> returns "MyHandler"
            ...     def execute_task(self, task, context): ...
        """
        # Default implementation: Use class name
        return self.__class__.__name__

    @abstractmethod
    def execute_task(self, task: Executable, context: ExecutionContext) -> Any:
        """Execute single task, store result in context, and return it when available.

        Abstract method - all handlers must implement execution strategy.

        Args:
            task: Executable task to execute
            context: Execution context

        Returns:
            Any result value produced by the task, when available. Handlers may
            return ``None`` when the underlying execution does not yield a value
            or when result retrieval is deferred.

        Usage Context:
            - Called by WorkflowEngine for individual task execution
            - Specified via @task(handler="docker") decorator
            - NOT called for tasks inside ParallelGroup (each task uses its own handler)
            - For ParallelGroup, only on_group_finished() is used

        Note:
            Implementation must call context.set_result(task_id, result) or
            context.set_result(task_id, exception) within the execution environment.

        Implementation Pattern:
            For execution handlers: Implement custom execution logic.
        """
        raise NotImplementedError
