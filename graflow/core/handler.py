"""Task execution handler base class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from graflow.core.context import ExecutionContext
from graflow.core.task import Executable

if TYPE_CHECKING:
    from graflow.core.handlers.group_policy import GroupExecutionPolicy


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

    TaskHandler has two roles used in different contexts:
    1. How individual tasks execute (execute_task) - via @task(handler="...")
    2. When parallel groups succeed/fail (on_group_finished) - via .with_execution(handler=...)

    Design:
    - execute_task() is abstract - all handlers must provide execution strategy
    - on_group_finished() has default implementation (strict mode)
    - get_name() has default implementation (class name)

    For policy handlers: Inherit from DirectTaskHandler to get execute_task() implementation.
    For execution handlers: Inherit from TaskHandler and implement execute_task().
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
    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """Execute single task and store result in context.

        Abstract method - all handlers must implement execution strategy.

        Args:
            task: Executable task to execute
            context: Execution context

        Usage Context:
            - Called by WorkflowEngine for individual task execution
            - Specified via @task(handler="docker") decorator
            - NOT called for tasks inside ParallelGroup (each task uses its own handler)
            - For ParallelGroup, only on_group_finished() is used

        Note:
            Implementation must call context.set_result(task_id, result) or
            context.set_result(task_id, exception) within the execution environment.

        Implementation Pattern:
            For policy handlers: Inherit from DirectTaskHandler instead.
            For execution handlers: Implement custom execution logic.
        """
        pass

    def set_group_policy(self, policy: 'GroupExecutionPolicy') -> None:
        """Assign a custom group execution policy for this handler."""
        self._group_policy = policy

    def get_group_policy(self) -> 'GroupExecutionPolicy':
        """Return the group execution policy for this handler."""
        from graflow.core.handlers.group_policy import StrictGroupPolicy

        policy = getattr(self, "_group_policy", None)
        if policy is None:
            policy = StrictGroupPolicy()
            self._group_policy = policy
        return policy

    def on_group_finished(
        self,
        group_id: str,
        tasks: Sequence[Executable],
        results: Dict[str, TaskResult],
        context: ExecutionContext
    ) -> None:
        """Handle parallel group execution results.

        Default: Strict mode - fail if any task fails.
        Override for custom success criteria.

        Args:
            group_id: Parallel group identifier
            tasks: List of tasks in the group
            results: Dict mapping task_id to TaskResult
            context: Execution context

        Raises:
            ParallelGroupError: If group execution should fail

        Usage Context:
            - Called by GroupExecutor after all parallel tasks complete
            - Specified via .with_execution(handler=PolicyHandler()) on ParallelGroup
            - NOT called for individual task execution
            - Use context.get_result(task_id) to access task result values if needed

        Note:
            Called after ALL tasks complete (success or failure).
            Handler decides whether to raise exception based on results.
        """
        policy = self.get_group_policy()
        policy.on_group_finished(group_id, tasks, results, context)
