"""Utilities for detecting unpicklable objects (locks, etc.) in object graphs.

This module provides utilities to analyze object graphs and find thread locks
and other unpicklable objects that would cause serialization failures.
"""

import logging
import threading
from collections import deque
from typing import Any, Iterator, List, Tuple

logger = logging.getLogger(__name__)

# Thread lock types (Python 3.11+)
# These are the actual types of threading.Lock() and threading.RLock()
LOCK_TYPES = (type(threading.Lock()), type(threading.RLock()))


def _iter_children(obj: Any) -> Iterator[Tuple[str, Any]]:  # noqa: PLR0912
    """Iterate over child objects that should be recursively explored.

    Args:
        obj: Object to explore

    Yields:
        Tuples of (path_fragment, child_object) where path_fragment is a string
        like "[key]", "[0]", or ".attr_name"
    """
    # Dict-like objects
    if isinstance(obj, dict):
        for k, v in obj.items():
            # Use repr for key to handle non-string keys
            yield f"[{k!r}]", v

    # List, tuple, set, frozenset
    elif isinstance(obj, list | tuple | set | frozenset):
        for i, v in enumerate(obj):
            yield f"[{i}]", v

    # Check for function closures (this is where locks often hide!)
    if callable(obj):
        # Check closure variables
        if hasattr(obj, "__closure__") and obj.__closure__:
            for i, cell in enumerate(obj.__closure__):
                try:
                    yield f".__closure__[{i}]", cell.cell_contents
                except ValueError:
                    # Empty cell
                    pass

        # Check __globals__ (functions carry their module globals)
        # This is critical because cloudpickle serializes referenced globals!
        if hasattr(obj, "__globals__"):
            for key, value in obj.__globals__.items():
                # Only check non-builtin globals to avoid infinite loops
                if not key.startswith("__") and key not in ("annotations", "functools", "uuid", "overload"):
                    yield f".__globals__[{key!r}]", value

    # Check for __wrapped__ attribute (common in decorators)
    if hasattr(obj, "__wrapped__"):
        yield ".__wrapped__", obj.__wrapped__  # type: ignore

    # Check for 'func' attribute (common in TaskWrapper and similar)
    if hasattr(obj, "func"):
        yield ".func", obj.func  # type: ignore

    # General objects with __dict__
    try:
        attrs = vars(obj)
    except TypeError:
        # Object doesn't have __dict__ (e.g., built-in types)
        return

    for name, v in attrs.items():
        yield f".{name}", v


def find_thread_locks(root: Any, max_depth: int = 8) -> List[Tuple[str, Any]]:
    """Find all thread lock objects in an object graph using BFS.

    Traverses the object graph starting from `root` and finds all instances
    of threading.Lock and threading.RLock.

    Args:
        root: Root object to start traversal from
        max_depth: Maximum depth to traverse (default: 8)

    Returns:
        List of (path, lock_object) tuples where path is a string like
        "graph.nodes[0].task._lock" indicating where the lock was found

    Example:
        >>> from graflow.core.graph import TaskGraph
        >>> graph = TaskGraph()
        >>> locks = find_thread_locks(graph)
        >>> for path, lock in locks:
        ...     print(f"Found lock at: {path}")
    """
    seen = set()
    queue = deque()
    queue.append((("root",), root))
    results = []

    while queue:
        path, obj = queue.popleft()

        # Skip if already visited (prevent infinite loops)
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        # Check if this object is a lock
        if isinstance(obj, LOCK_TYPES):
            path_str = "".join(path)
            results.append((path_str, obj))
            # Don't traverse into lock internals
            continue

        # Stop if we've reached max depth
        if len(path) > max_depth:
            continue

        # Add children to queue
        try:
            for fragment, child in _iter_children(obj):
                queue.append(((*path, fragment), child))
        except Exception as e:
            # Log but don't fail if we can't iterate children
            logger.debug(f"Error iterating children of {path}: {e}")
            continue

    return results


def debug_taskgraph_for_locks(graph: Any) -> None:
    """Analyze a TaskGraph for thread locks and log findings.

    This is a debug utility to help identify unpicklable lock objects
    that would cause serialization failures.

    Args:
        graph: TaskGraph object to analyze

    Example:
        >>> from graflow.core.graph import TaskGraph
        >>> graph = TaskGraph()
        >>> debug_taskgraph_for_locks(graph)
    """
    logger.info("Analyzing TaskGraph for thread locks...")

    # Try progressively deeper searches
    for depth in [10, 20, 30]:
        try:
            locks = find_thread_locks(graph, max_depth=depth)
            if locks:
                logger.warning(f"Found {len(locks)} thread lock(s) at depth={depth}")
                break
            else:
                logger.info(f"No locks found at depth={depth}, trying deeper...")
        except Exception as e:
            logger.error(f"Error analyzing TaskGraph at depth={depth}: {e}", exc_info=True)
            return
    else:
        logger.info("No thread locks found in TaskGraph (searched up to depth=30)")
        return

    # Found locks - log detailed information
    logger.warning(f"Found {len(locks)} thread lock(s) in TaskGraph:")
    for path, lock in locks[:10]:  # Show first 10
        lock_type = type(lock).__name__
        msg = f"  Lock at: {path}"
        msg += f"\n    Type: {lock_type}"
        msg += f"\n    Object: {lock!r}"
        logger.warning(msg)

    if len(locks) > 10:
        logger.warning(f"  ... and {len(locks) - 10} more locks")

    # Additional guidance
    logger.warning("Serialization will fail due to these locks")
    logger.info("Suggestion: Ensure __getstate__/__setstate__ methods exclude lock objects")


def find_unpicklable_objects(root: Any, max_depth: int = 8) -> List[Tuple[str, Any, str]]:
    """Find common unpicklable objects in an object graph.

    This is a more general version of find_thread_locks that also looks for
    other common unpicklable types.

    Args:
        root: Root object to start traversal from
        max_depth: Maximum depth to traverse

    Returns:
        List of (path, object, reason) tuples
    """
    seen = set()
    queue = deque()
    queue.append((("root",), root))
    results = []

    # Types that are commonly unpicklable
    import types

    unpicklable_types = [
        (LOCK_TYPES, "threading lock"),
        (type(lambda: None), "lambda function"),
        (types.ModuleType, "module"),
        (type(open(__file__)), "file object"),
    ]

    while queue:
        path, obj = queue.popleft()

        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        # Check if this object is unpicklable
        for type_or_types, reason in unpicklable_types:
            if isinstance(obj, type_or_types):
                path_str = "".join(path)
                results.append((path_str, obj, reason))
                # Don't traverse into unpicklable objects
                break
        else:
            # Only traverse if not unpicklable
            if len(path) <= max_depth:
                try:
                    for fragment, child in _iter_children(obj):
                        queue.append(((*path, fragment), child))
                except Exception:
                    pass

    return results
