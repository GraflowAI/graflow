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
def task(id_or_func: F) -> TaskWrapper: ... # type: ignore
# Usage: @task (without parentheses, directly on function)

@overload
def task(id_or_func: str, *, inject_context: bool = False) -> Callable[[F], TaskWrapper]: ... # type: ignore
# Usage: @task("custom_id") or @task("custom_id", inject_context=True)

@overload
def task(*, id: Optional[str] = None, inject_context: bool = False) -> Callable[[F], TaskWrapper]: ... # type: ignore
# Usage: @task() or @task(id="custom_id") or @task(inject_context=True)

def task(
    id_or_func: Optional[F] | str | None = None, *, id: Optional[str] = None, inject_context: bool = False
) -> TaskWrapper | Callable[[F], TaskWrapper]:
    """Decorator to convert a function into a Task object.

    Can be used as:
    - @task
    - @task()
    - @task("custom_id")
    - @task("custom_id", inject_context=True)
    - @task(id="custom_id")
    - @task(inject_context=True)

    Args:
        id_or_func: The function to decorate (when used without parentheses) or task ID string
        id: Optional custom id for the task (keyword argument)
        inject_context: If True, automatically inject ExecutionContext as first parameter

    Returns:
        TaskWrapper instance or decorator function
    """

    # Handle @task("task_id") and @task("task_id", inject_context=True) syntax
    if isinstance(id_or_func, str):
        def string_decorator(f: F) -> TaskWrapper:
            return task(f, id=id_or_func, inject_context=inject_context)  # type: ignore
        return string_decorator

    def decorator(f: F) -> TaskWrapper:
        # Get task id. Use provided id, or function name, or random UUID.
        task_id = id if id is not None else getattr(f, '__name__', None)
        if task_id is None:
            task_id = str(uuid.uuid4().int)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        # Create TaskWrapper instance
        from .task import TaskWrapper  # Import here to avoid circular imports
        task_obj = TaskWrapper(task_id, wrapper, inject_context=inject_context)

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

    if callable(id_or_func):
        # If the decorator is used without parentheses, apply it directly
        return decorator(id_or_func)

    # Handle @task() or @task(id="...")
    return decorator
