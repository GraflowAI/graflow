"""Decorators for graflow tasks."""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, TypeVar, overload

if TYPE_CHECKING:
    from .task import TaskWrapper

F = TypeVar('F', bound=Callable[..., Any])

@overload
def task(func: F) -> TaskWrapper: ... # type: ignore

@overload
def task(func: None = None, *, id: Optional[str] = None) -> Callable[[F], TaskWrapper]: ... # type: ignore

def task(
    func: Optional[F] = None, *, id: Optional[str] = None
) -> TaskWrapper | Callable[[F], TaskWrapper]:
    """Decorator to convert a function into a Task object.

    Can be used as:
    - @task
    - @task()
    - @task(id="custom_id")

    Args:
        func: The function to decorate (when used without parentheses)
        id: Optional custom id for the task

    Returns:
        TaskWrapper instance or decorator function
    """

    def decorator(f: F) -> TaskWrapper:
        # Get task id. Use random UUID if not provided.
        task_id = id if id is not None else getattr(f, '__name__', None)
        if task_id is None:
            task_id = str(uuid.uuid4())

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        # Create TaskWrapper instance
        try:
            from .task import TaskWrapper  # Import here to avoid circular imports  # noqa: PLC0415
            task_obj = TaskWrapper(task_id, wrapper)
        except Exception as e:
            raise RuntimeError(f"Failed to create TaskWrapper for {task_id}: {e}") from e

        # Copy original function attributes to ensure compatibility
        try:
            task_obj.__name__ = f.__name__
            task_obj.__doc__ = f.__doc__
            # Only set __module__ if it's a string
            module = getattr(f, '__module__', None)
            if isinstance(module, str):
                task_obj.__module__ = module
            task_obj.__qualname__ = getattr(f, '__qualname__', f.__name__)
            task_obj.__annotations__ = getattr(f, '__annotations__', {})
        except (AttributeError, TypeError):
            # Some attributes might not be settable, continue gracefully
            pass

        return task_obj

    if func is not None:
        # If the decorator is used without parentheses, apply it directly
        return decorator(func)

    # Handle @task() or @task(id="...")
    return decorator
