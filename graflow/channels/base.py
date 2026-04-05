"""Base channel interface for extensibility."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, Union


class Channel(ABC):
    """Abstract base class for all channel implementations."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store data in the channel."""
        pass

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data from the channel."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key from the channel."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the channel."""
        pass

    @abstractmethod
    def keys(self) -> List[str]:
        """Get all keys in the channel."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the channel."""
        pass

    @abstractmethod
    def append(self, key: str, value: Any, ttl: Optional[int] = None) -> int:
        """Append value to a list stored at key.

        Args:
            key: The key identifying the list
            value: Value to append to the list
            ttl: Optional time-to-live in seconds for the key

        Returns:
            Length of the list after append
        """
        pass

    @abstractmethod
    def prepend(self, key: str, value: Any, ttl: Optional[int] = None) -> int:
        """Prepend value to the head of a list stored at key.

        Args:
            key: The key identifying the list
            value: Value to prepend to the list
            ttl: Optional time-to-live in seconds for the key

        Returns:
            Length of the list after prepend
        """
        pass

    @abstractmethod
    def atomic_add(self, key: str, amount: Union[int, float] = 1) -> Union[int, float]:
        """Atomically add *amount* to the numeric value stored at *key*.

        If *key* does not exist, it is initialised to 0 before the addition.
        Negative *amount* is allowed (decrement).

        Args:
            key: The key identifying the numeric value.
            amount: The value to add (default 1). May be negative.

        Returns:
            The value after the addition.

        Raises:
            TypeError: If the existing value is not numeric.
        """
        pass

    @contextmanager
    def lock(self, key: str, timeout: float = 10.0) -> Iterator[None]:
        """Acquire an advisory lock scoped to *key* for compound operations.

        Usage::

            with channel.lock("counter"):
                val = channel.get("counter")
                channel.set("counter", val * 2 if val > 0 else 0)

        The lock is *advisory* — regular ``get``/``set`` calls do **not**
        acquire it automatically.  It exists for task authors who need to
        protect read-modify-write sequences that cannot be expressed with
        ``atomic_add()``.

        The default implementation is a **no-op** (yields immediately).
        This is appropriate for backends where atomicity is guaranteed by
        the server (e.g. Redis commands are serialised server-side).
        Subclasses that need client-side locking (e.g. ``MemoryChannel``
        under multi-threading) should override this method.

        Args:
            key: Logical key to lock on (does not need to correspond to a
                 stored key).
            timeout: Maximum seconds to wait for the lock.

        Yields:
            None — the lock is held for the duration of the ``with`` block.
        """
        yield
