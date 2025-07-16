"""
graflow.exceptions
=================
This module defines custom exceptions for the Graflow library.
"""

class GraflowError(Exception):
    """Base exception class for Graflow errors."""
    pass

class GraphError(GraflowError):
    """Exception raised for errors related to the task graph."""
    pass

class TaskError(GraflowError):
    """Exception raised for errors related to tasks."""
    pass
