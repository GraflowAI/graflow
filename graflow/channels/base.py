"""Base channel interface for extensibility."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional


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
