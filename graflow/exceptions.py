"""
graflow.exceptions
=================
This module defines custom exceptions for the Graflow library.
"""

from typing import Optional, Sequence


class GraflowError(Exception):
    """Base exception class for Graflow errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{super().__str__()} (caused by {self.cause})"
        return super().__str__()


class GraflowRuntimeError(GraflowError):
    """Exception raised for runtime errors in Graflow."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, cause)


class CycleLimitExceededError(GraflowRuntimeError):
    """Exception raised when the cycle limit is exceeded during execution."""

    def __init__(self, task_id: str, cycle_count: int, max_cycles: int):
        super().__init__(f"Cycle limit exceeded for task {task_id}: {cycle_count}/{max_cycles} cycles")
        self.task_id = task_id
        self.cycle_count = cycle_count
        self.max_cycles = max_cycles

    def __str__(self) -> str:
        return f"CycleLimitExceededError(task_id={self.task_id}, cycle_count={self.cycle_count}, max_cycles={self.max_cycles})"


class TaskError(GraflowRuntimeError):
    """Exception raised for errors related to tasks."""

    def __init__(self, task_id: str, message: str, cause: Optional[Exception] = None):
        super().__init__(f"Error in task '{task_id}': {message}", cause)
        self._task_id = task_id

    @property
    def task_id(self) -> str:
        """Return the ID of the task that caused the error."""
        return self._task_id


class GraphCompilationError(GraflowError):
    """Exception raised for errors during graph compilation."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, cause)


class DuplicateTaskError(GraphCompilationError):
    """Exception raised when a task with the same ID already exists."""

    def __init__(self, task_id: str):
        super().__init__(f"Duplicate task ID: {task_id}")
        self.task_id = task_id


class ConfigError(GraflowError):
    """Exception raised for configuration errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message, cause)


class ParallelGroupError(GraflowRuntimeError):
    """Exception raised when parallel group execution fails.

    This is a subclass of GraflowRuntimeError, so it's compatible
    with existing exception handling while providing specific
    information about parallel group failures.

    Attributes:
        group_id: Parallel group identifier
        failed_tasks: List of (task_id, error_message) tuples where error_message may be None
        successful_tasks: List of successful task IDs
    """

    def __init__(
        self,
        message: str,
        group_id: str,
        failed_tasks: Sequence[tuple[str, str | None]],
        successful_tasks: Sequence[str],
    ):
        super().__init__(message)
        self.group_id = group_id
        self.failed_tasks = failed_tasks
        self.successful_tasks = successful_tasks


class GraflowWorkflowCanceledError(GraflowRuntimeError):
    """Exception raised when a workflow is explicitly canceled by a task.

    This allows tasks to cancel the entire workflow execution, skipping
    all remaining tasks and propagating the cancellation error.

    Attributes:
        message: Cancellation reason/message
        task_id: ID of the task that requested cancellation
    """

    def __init__(self, message: str, task_id: Optional[str] = None):
        super().__init__(message)
        self.task_id = task_id

    def __str__(self) -> str:
        if self.task_id:
            return f"Workflow canceled by task '{self.task_id}': {self.args[0]}"
        return f"Workflow canceled: {self.args[0]}"


def as_runtime_error(ex: Exception) -> GraflowRuntimeError:
    """Wrap a generic exception into a GraflowRuntimeError."""
    if isinstance(ex, GraflowRuntimeError):
        return ex
    else:
        # Wrap any other exception into a GraflowRuntimeError
        return GraflowRuntimeError(f"An error occurred: {str(ex)}", cause=ex)
