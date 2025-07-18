"""Channel system for inter-task communication."""

from .base import Channel
from .factory import ChannelFactory
from .memory import MemoryChannel

try:
    from .redis import RedisChannel
    __all__ = ["Channel", "ChannelFactory", "MemoryChannel", "RedisChannel"]
except ImportError:
    __all__ = ["Channel", "ChannelFactory", "MemoryChannel"]
