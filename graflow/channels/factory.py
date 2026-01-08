"""Factory for creating channel instances based on backend type."""

from __future__ import annotations

from typing import ClassVar, Dict, Type

from graflow.channels.base import Channel
from graflow.channels.memory_channel import MemoryChannel
from graflow.exceptions import ConfigError

try:
    from graflow.channels.redis_channel import RedisChannel

    REDIS_AVAILABLE = True
except ImportError:
    RedisChannel = None
    REDIS_AVAILABLE = False


class ChannelFactory:
    """Factory for creating channel instances based on backend type."""

    _backends: ClassVar[Dict[str, Type[Channel]]] = {
        "memory": MemoryChannel,
    }

    if REDIS_AVAILABLE and RedisChannel is not None:
        _backends["redis"] = RedisChannel

    @classmethod
    def create_channel(cls, backend: str, name: str, **kwargs) -> Channel:
        """Create a channel instance based on backend type.

        Args:
            backend: Channel backend type ('memory' or 'redis')
            name: Channel name
            **kwargs: Additional arguments for the channel constructor

        Returns:
            Channel instance

        Raises:
            ConfigError: If backend is not supported
        """
        if backend not in cls._backends:
            available_backends = list(cls._backends.keys())
            raise ConfigError(f"Unsupported backend '{backend}'. Available backends: {available_backends}")

        channel_class = cls._backends[backend]
        return channel_class(name, **kwargs)

    @classmethod
    def register_backend(cls, name: str, channel_class: Type[Channel]) -> None:
        """Register a new channel backend.

        Args:
            name: Backend name
            channel_class: Channel class that extends Channel
        """
        if not issubclass(channel_class, Channel):
            raise TypeError(f"Channel class must extend Channel, got {channel_class}")

        cls._backends[name] = channel_class

    @classmethod
    def get_available_backends(cls) -> list[str]:
        """Get list of available channel backends."""
        return list(cls._backends.keys())

    @classmethod
    def is_backend_available(cls, backend: str) -> bool:
        """Check if a backend is available."""
        return backend in cls._backends
