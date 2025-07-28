"""
graflow.exceptions
=================
This module defines custom exceptions for the Graflow library.
"""

from typing import Optional


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

def as_runtime_error(ex: Exception) -> GraflowRuntimeError:
    """Wrap a generic exception into a GraflowRuntimeError."""
    if isinstance(ex, GraflowRuntimeError):
        return ex
    else:
        # Wrap any other exception into a GraflowRuntimeError
        return GraflowRuntimeError(f"An error occurred: {str(ex)}", cause=ex)
