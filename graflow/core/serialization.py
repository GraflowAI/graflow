"""Serialization utilities using cloudpickle for robust pickling.

This module provides a unified interface for serialization that uses cloudpickle
for better support of lambdas, closures, and dynamically generated functions.
Falls back to standard pickle if cloudpickle is not available.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import cloudpickle

    HAS_CLOUDPICKLE = True
except ImportError:
    import pickle as cloudpickle  # type: ignore[no-redef]

    HAS_CLOUDPICKLE = False


def dumps(obj: Any) -> bytes:
    """Serialize object using cloudpickle (or pickle as fallback).

    Args:
        obj: Object to serialize

    Returns:
        Serialized bytes

    Raises:
        TypeError: If object contains unpicklable objects (e.g., locks)

    Note:
        Uses cloudpickle for better support of lambdas and closures.
        Falls back to standard pickle if cloudpickle is not available.

        If serialization fails with TypeError, this function will attempt
        to analyze the object for common unpicklable objects (thread locks)
        and provide detailed debug information before re-raising the error.
    """
    try:
        return cloudpickle.dumps(obj)
    except TypeError as e:
        # Serialization failed - analyze object for unpicklable components
        logger.error(
            f"Failed to serialize object of type {type(obj).__name__}. Serialization failed with TypeError: {e}"
        )

        # Try to detect TaskGraph objects for lock analysis
        from graflow.core.graph import TaskGraph

        if isinstance(obj, TaskGraph):
            logger.info("Analyzing TaskGraph for thread locks...")
            try:
                from graflow.debug.find_locks import debug_taskgraph_for_locks

                debug_taskgraph_for_locks(obj)
            except Exception as debug_err:
                logger.warning(f"Failed to analyze object for locks: {debug_err}")

        # Re-raise original exception
        raise


def loads(data: bytes) -> Any:
    """Deserialize object using cloudpickle (or pickle as fallback).

    Args:
        data: Serialized bytes

    Returns:
        Deserialized object
    """
    return cloudpickle.loads(data)


def dump(obj: Any, file: Any) -> None:
    """Serialize object to file using cloudpickle.

    Args:
        obj: Object to serialize
        file: File object to write to
    """
    cloudpickle.dump(obj, file)


def load(file: Any) -> Any:
    """Deserialize object from file using cloudpickle.

    Args:
        file: File object to read from

    Returns:
        Deserialized object
    """
    return cloudpickle.load(file)
