"""Serialization utilities using cloudpickle for robust pickling.

This module provides a unified interface for serialization that uses cloudpickle
for better support of lambdas, closures, and dynamically generated functions.
Falls back to standard pickle if cloudpickle is not available.
"""

from typing import Any

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

    Note:
        Uses cloudpickle for better support of lambdas and closures.
        Falls back to standard pickle if cloudpickle is not available.
    """
    return cloudpickle.dumps(obj)


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
