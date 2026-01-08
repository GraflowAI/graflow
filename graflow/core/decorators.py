"""Decorators for graflow tasks."""

from __future__ import annotations

import functools
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, TypeVar, overload

if TYPE_CHECKING:
    from .task import TaskWrapper

F = TypeVar("F", bound=Callable[..., Any])


@overload
def task(id_or_func: F) -> TaskWrapper: ...  # type: ignore


# Usage: @task (without parentheses, directly on function)


@overload
def task(
    id_or_func: str,
    *,
    inject_context: bool = False,
    inject_llm_client: bool = False,
    inject_llm_agent: Optional[str] = None,
    handler: Optional[str] = None,
    resolve_keyword_args: bool = True,
) -> Callable[[F], TaskWrapper]: ...  # type: ignore


# Usage: @task("custom_id") or @task("custom_id", inject_context=True, handler="docker")


@overload
def task(
    *,
    id: Optional[str] = None,
    inject_context: bool = False,
    inject_llm_client: bool = False,
    inject_llm_agent: Optional[str] = None,
    handler: Optional[str] = None,
    resolve_keyword_args: bool = True,
) -> Callable[[F], TaskWrapper]: ...  # type: ignore


# Usage: @task() or @task(id="custom_id") or @task(inject_context=True) or @task(handler="docker")


def task(
    id_or_func: Optional[F] | str | None = None,
    *,
    id: Optional[str] = None,
    inject_context: bool = False,
    inject_llm_client: bool = False,
    inject_llm_agent: Optional[str] = None,
    handler: Optional[str] = None,
    resolve_keyword_args: bool = True,
) -> TaskWrapper | Callable[[F], TaskWrapper]:
    """Decorator to convert a function into a Task object.

    Can be used as:
    - @task
    - @task()
    - @task("custom_id")
    - @task("custom_id", inject_context=True)
    - @task(id="custom_id")
    - @task(inject_context=True)
    - @task(inject_llm_client=True)
    - @task(inject_llm_agent="supervisor")
    - @task(handler="docker")
    - @task("custom_id", handler="docker")

    Args:
        id_or_func: The function to decorate (when used without parentheses) or task ID string
        id: Optional custom id for the task (keyword argument)
        inject_context: If True, automatically inject TaskExecutionContext as first parameter
        inject_llm_client: If True, automatically inject shared LLMClient instance from
                          ExecutionContext as first parameter
        inject_llm_agent: Agent name string to inject LLMAgent as first parameter
                         (agent must be registered in ExecutionContext)
        handler: Execution handler type ("direct", "docker", or custom)
        resolve_keyword_args: If True (default), automatically resolve function keyword arguments
                             from channel by matching parameter names to channel keys

    Returns:
        TaskWrapper instance or decorator function

    Example:
        ```python
        @task(inject_llm_client=True)
        def my_task(llm: LLMClient, data: str) -> str:
            # Use default model
            result1 = llm.completion([{"role": "user", "content": data}])

            # Or override model per call
            result2 = llm.completion(
                [{"role": "user", "content": data}],
                model="gpt-4o"
            )
            return result1, result2
        ```
    """

    # Handle @task("task_id") and @task("task_id", inject_context=True) syntax
    if isinstance(id_or_func, str):

        def string_decorator(f: F) -> TaskWrapper:
            return task(
                f,
                id=id_or_func,
                inject_context=inject_context,
                inject_llm_client=inject_llm_client,
                inject_llm_agent=inject_llm_agent,
                handler=handler,
                resolve_keyword_args=resolve_keyword_args,
            )  # type: ignore

        return string_decorator

    def decorator(f: F) -> TaskWrapper:
        # Get task id. Use provided id, or function name, or random UUID.
        task_id = id if id is not None else getattr(f, "__name__", None)
        if task_id is None:
            task_id = str(uuid.uuid4().int)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        # Drop typing.overload from decorator globals to avoid leaking typing module locks via __globals__
        wrapper.__globals__.pop("overload", None)  # type: ignore

        # Create TaskWrapper instance
        from .task import TaskWrapper  # Import here to avoid circular imports

        task_obj = TaskWrapper(
            task_id,
            wrapper,
            inject_context=inject_context,
            inject_llm_client=inject_llm_client,
            inject_llm_agent=inject_llm_agent,
            handler_type=handler,
            resolve_keyword_args=resolve_keyword_args,
        )

        # Copy original function attributes to ensure compatibility
        try:
            task_obj.__name__ = f.__name__
            task_obj.__doc__ = f.__doc__
            # Only set __module__ if it's a string
            module = getattr(f, "__module__", None)
            if isinstance(module, str):
                task_obj.__module__ = module
            task_obj.__qualname__ = getattr(f, "__qualname__", f.__name__)
            task_obj.__annotations__ = getattr(f, "__annotations__", {})
        except (AttributeError, TypeError):
            # Some attributes might not be settable, continue gracefully
            pass

        return task_obj

    if callable(id_or_func):
        # If the decorator is used without parentheses, apply it directly
        return decorator(id_or_func)

    # Handle @task() or @task(id="...")
    return decorator
