"""Task execution handler base class."""

from abc import ABC, abstractmethod

from graflow.core.context import ExecutionContext
from graflow.core.task import Executable


class TaskHandler(ABC):
    """Base class for task execution handlers.

    TaskHandler defines the interface for executing tasks in different environments.
    Each handler implementation is responsible for:
    - Executing the task in its specific environment
    - Calling context.set_result(task_id, result) within the execution environment
    - Handling exceptions and storing them via context.set_result(task_id, exception)

    For remote execution environments (e.g., Docker, Kubernetes), implementations
    should ensure that context.set_result() is called inside the remote environment.
    """

    @abstractmethod
    def execute_task(self, task: Executable, context: ExecutionContext) -> None:
        """Execute task and store result in context.

        Args:
            task: Executable task to execute
            context: Execution context

        Note:
            Implementation must call context.set_result(task_id, result) or
            context.set_result(task_id, exception) within the execution environment.

            For remote execution (e.g., Docker, Kubernetes), the implementation
            should ensure context.set_result() is called inside the remote environment.
        """
        pass
