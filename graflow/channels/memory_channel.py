"""Memory-based channel implementation for inter-task communication."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Union

from graflow.channels.base import Channel


class MemoryChannel(Channel):
    """Memory-based channel implementation for inter-task communication."""

    def __init__(self, name: str, **kwargs):
        """Initialize memory channel."""
        super().__init__(name)
        self.data: Dict[str, Any] = {}
        self.ttl_data: Dict[str, float] = {}
        self._key_locks: Dict[str, threading.RLock] = {}
        self._key_locks_guard = threading.Lock()  # protects _key_locks dict itself

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude unpicklable lock objects during serialization."""
        state = self.__dict__.copy()
        del state["_key_locks"]
        del state["_key_locks_guard"]
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Recreate lock objects after deserialization."""
        self.__dict__.update(state)  # type: ignore[assignment]
        self._key_locks = {}
        self._key_locks_guard = threading.Lock()

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store data in the channel."""
        self.data[key] = value
        if ttl is not None:
            self.ttl_data[key] = time.time() + ttl

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data from the channel."""
        self._cleanup_expired(key)
        return self.data.get(key, default)

    def delete(self, key: str) -> bool:
        """Delete a key from the channel."""
        existed = key in self.data
        if existed:
            del self.data[key]
        if key in self.ttl_data:
            del self.ttl_data[key]
        return existed

    def exists(self, key: str) -> bool:
        """Check if a key exists in the channel."""
        self._cleanup_expired(key)
        return key in self.data

    def keys(self) -> List[str]:
        """Get all keys in the channel."""
        # Clean up expired keys first
        expired_keys = []
        current_time = time.time()
        for key, expire_time in self.ttl_data.items():
            if current_time > expire_time:
                expired_keys.append(key)

        for key in expired_keys:
            self.delete(key)

        return list(self.data.keys())

    def clear(self) -> None:
        """Clear all data from the channel."""
        self.data.clear()
        self.ttl_data.clear()

    def _cleanup_expired(self, key: str) -> None:
        """Remove expired key if TTL has passed."""
        if key in self.ttl_data:
            if time.time() > self.ttl_data[key]:
                self.delete(key)

    def append(self, key: str, value: Any, ttl: Optional[int] = None) -> int:
        """Append value to a list stored at key.

        Args:
            key: The key identifying the list
            value: Value to append to the list
            ttl: Optional time-to-live in seconds for the key

        Returns:
            Length of the list after append
        """
        self._cleanup_expired(key)

        # Initialize list if key doesn't exist
        if key not in self.data:
            self.data[key] = []
        elif not isinstance(self.data[key], list):
            raise TypeError(f"Key '{key}' exists but is not a list")

        # Append to the list
        self.data[key].append(value)

        # Set TTL if specified
        if ttl is not None:
            self.ttl_data[key] = time.time() + ttl

        return len(self.data[key])

    def prepend(self, key: str, value: Any, ttl: Optional[int] = None) -> int:
        """Prepend value to the head of a list stored at key.

        Args:
            key: The key identifying the list
            value: Value to prepend to the list
            ttl: Optional time-to-live in seconds for the key

        Returns:
            Length of the list after prepend
        """
        self._cleanup_expired(key)

        # Initialize list if key doesn't exist
        if key not in self.data:
            self.data[key] = []
        elif not isinstance(self.data[key], list):
            raise TypeError(f"Key '{key}' exists but is not a list")

        # Prepend to the head of the list
        self.data[key].insert(0, value)

        # Set TTL if specified
        if ttl is not None:
            self.ttl_data[key] = time.time() + ttl

        return len(self.data[key])

    def atomic_add(self, key: str, amount: Union[int, float] = 1) -> Union[int, float]:
        """Atomically add *amount* to the numeric value stored at *key*.

        Thread-safe: acquires a per-key lock so concurrent calls do not lose
        updates.

        Args:
            key: The key identifying the numeric value.
            amount: The value to add (default 1). May be negative.

        Returns:
            The value after the addition.

        Raises:
            TypeError: If the existing value is not numeric.
        """
        with self._get_key_lock(key):
            self._cleanup_expired(key)
            current = self.data.get(key, 0)
            if not isinstance(current, int | float):
                raise TypeError(f"Key '{key}' holds {type(current).__name__}, expected int or float")
            new_value = current + amount
            self.data[key] = new_value
            return new_value

    # -- advisory locking for compound operations --

    def _get_key_lock(self, key: str) -> threading.RLock:
        """Return (or create) the RLock associated with *key*."""
        if key not in self._key_locks:
            with self._key_locks_guard:
                # Double-checked locking
                if key not in self._key_locks:
                    self._key_locks[key] = threading.RLock()
        return self._key_locks[key]

    @contextmanager
    def lock(self, key: str, timeout: float = 10.0) -> Iterator[None]:
        """Acquire an advisory per-key lock for compound read-modify-write.

        Args:
            key: Logical key to lock on.
            timeout: Maximum seconds to wait for the lock.

        Raises:
            TimeoutError: If the lock cannot be acquired within *timeout*.
        """
        rlock = self._get_key_lock(key)
        acquired = rlock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Could not acquire lock for key '{key}' within {timeout}s")
        try:
            yield
        finally:
            rlock.release()
