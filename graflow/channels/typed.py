"""Typed channel wrapper for type-safe inter-task communication."""

from __future__ import annotations

from typing import Any, ClassVar, Generic, Type, TypeVar, get_type_hints

from graflow.channels.base import Channel
from graflow.exceptions import ConfigError


def _is_typed_dict(cls: type) -> bool:
    """Check if a class is a TypedDict."""
    return hasattr(cls, "__annotations__") and hasattr(cls, "__total__")


def _validate_typed_dict(data: Any, typed_dict_class: Type) -> bool:
    """Validate that data conforms to TypedDict schema."""
    if not isinstance(data, dict):
        return False

    if not _is_typed_dict(typed_dict_class):
        return False

    annotations = get_type_hints(typed_dict_class)
    required_keys = set(annotations.keys())

    # Check if all required keys are present
    if hasattr(typed_dict_class, "__required_keys__"):
        required_keys = typed_dict_class.__required_keys__
    elif hasattr(typed_dict_class, "__total__") and not typed_dict_class.__total__:
        # If __total__ is False, no keys are required
        required_keys = set()

    data_keys = set(data.keys())

    # Check required keys
    if not required_keys.issubset(data_keys):
        return False

    # Check that no extra keys are present
    allowed_keys = set(annotations.keys())
    if not data_keys.issubset(allowed_keys):
        return False

    # Basic type checking for present keys
    for key, value in data.items():
        if key in annotations:
            expected_type = annotations[key]
            # Skip complex type checking for now, just do basic validation
            if expected_type in (str, int, float, bool, list, dict):
                if not isinstance(value, expected_type):
                    return False

    return True


T = TypeVar("T")


class TypedChannel(Channel, Generic[T]):
    """Type-safe wrapper around Channel for structured message passing."""

    def __init__(self, channel: Channel, message_type: Type[T]):
        """Initialize typed channel.

        Args:
            channel: Underlying channel implementation
            message_type: TypedDict class defining message structure
        """
        self._channel = channel
        self._message_type = message_type

        if not _is_typed_dict(message_type):
            raise ValueError(f"message_type must be a TypedDict, got {message_type}")

    @property
    def name(self) -> str:
        """Get channel name."""
        return self._channel.name

    @property
    def message_type(self) -> Type[T]:
        """Get message type."""
        return self._message_type

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store data in the channel."""
        if not _validate_typed_dict(value, self._message_type):
            raise TypeError(f"Value does not conform to {self._message_type.__name__}")

        self._channel.set(key, value, ttl)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data from the channel."""
        data = self._channel.get(key, default)

        if data is None:
            return None

        if _validate_typed_dict(data, self._message_type):
            return data

        return None

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        return self._channel.exists(key)

    def delete(self, key: str) -> bool:
        """Delete a key."""
        return self._channel.delete(key)

    def keys(self) -> list[str]:
        """Get all keys."""
        return self._channel.keys()

    def clear(self) -> None:
        """Clear all data."""
        self._channel.clear()

    def append(self, key: str, value: Any, ttl: int | None = None) -> int:
        """Append value to a list stored at key.

        Note: For list operations, type validation is skipped.

        Args:
            key: The key identifying the list
            value: Value to append to the list
            ttl: Optional time-to-live in seconds for the key

        Returns:
            Length of the list after append
        """
        return self._channel.append(key, value, ttl)

    def prepend(self, key: str, value: Any, ttl: int | None = None) -> int:
        """Prepend value to the head of a list stored at key.

        Note: For list operations, type validation is skipped.

        Args:
            key: The key identifying the list
            value: Value to prepend to the list
            ttl: Optional time-to-live in seconds for the key

        Returns:
            Length of the list after prepend
        """
        return self._channel.prepend(key, value, ttl)

    def send(self, key: str, message: T, ttl: int | None = None) -> None:
        """Send a typed message.

        Args:
            key: Message key
            message: Message data conforming to type T
            ttl: Optional time-to-live in seconds

        Raises:
            TypeError: If message doesn't conform to expected type
        """
        if not _validate_typed_dict(message, self._message_type):
            raise TypeError(f"Message does not conform to {self._message_type.__name__}")

        self._channel.set(key, message, ttl)

    def receive(self, key: str, default: T | None = None) -> T | None:
        """Receive a typed message.

        Args:
            key: Message key
            default: Default value if key not found

        Returns:
            Message data or None if not found/invalid
        """
        data = self._channel.get(key, default)

        if data is None:
            return None

        if _validate_typed_dict(data, self._message_type):
            return data

        return None


class ChannelTypeRegistry:
    """Registry for commonly used message types."""

    _types: ClassVar[dict[str, Type]] = {}

    @classmethod
    def register(cls, name: str, message_type: Type) -> None:
        """Register a message type."""
        if not _is_typed_dict(message_type):
            raise ValueError(f"message_type must be a TypedDict, got {message_type}")
        cls._types[name] = message_type

    @classmethod
    def get(cls, name: str) -> Type | None:
        """Get a registered message type."""
        return cls._types.get(name)

    @classmethod
    def create_channel(cls, channel: Channel, type_name: str) -> TypedChannel:
        """Create a typed channel from registered type."""
        message_type = cls.get(type_name)
        if message_type is None:
            raise ConfigError(f"Message type '{type_name}' not registered")
        return TypedChannel(channel, message_type)
