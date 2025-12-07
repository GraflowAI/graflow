"""Memory-based channel implementation for inter-task communication."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from graflow.channels.base import Channel


class MemoryChannel(Channel):
    """Memory-based channel implementation for inter-task communication."""

    def __init__(self, name: str, **kwargs):
        """Initialize memory channel."""
        super().__init__(name)
        self.data: Dict[str, Any] = {}
        self.ttl_data: Dict[str, float] = {}

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
