"""Redis-based channel implementation for inter-task communication."""

from __future__ import annotations

import json
from typing import Any, List, Optional, Union, cast

from .base import Channel

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None


class RedisChannel(Channel):
    """Redis-based channel implementation for inter-task communication."""

    def __init__(self, name: str, host: str = "localhost", port: int = 6379, db: int = 0, **kwargs):
        """Initialize Redis channel."""
        super().__init__(name)
        if redis is None:
            raise ImportError("redis package is required for RedisChannel")

        self.redis_client: Redis = redis.Redis(host=host, port=port, db=db, decode_responses=True, **kwargs)
        self.key_prefix = f"graflow:channel:{name}:"

    def _get_key(self, key: str) -> str:
        """Get the full Redis key with prefix."""
        return f"{self.key_prefix}{key}"

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store data in the channel."""
        redis_key = self._get_key(key)
        serialized_value = json.dumps(value)

        if ttl is not None:
            self.redis_client.setex(redis_key, ttl, serialized_value)
        else:
            self.redis_client.set(redis_key, serialized_value)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data from the channel."""
        redis_key = self._get_key(key)
        value = self.redis_client.get(redis_key)

        if value is None:
            return default

        try:
            # Redis with decode_responses=True returns str, but we cast to be safe
            value_str = value if isinstance(value, str) else str(value)
            return json.loads(value_str)
        except (json.JSONDecodeError, TypeError):
            return default

    def delete(self, key: str) -> bool:
        """Delete a key from the channel."""
        redis_key = self._get_key(key)
        deleted_count = self.redis_client.delete(redis_key)
        # Redis delete returns the number of keys deleted
        return cast(int, deleted_count) > 0

    def exists(self, key: str) -> bool:
        """Check if a key exists in the channel."""
        redis_key = self._get_key(key)
        exists_result = self.redis_client.exists(redis_key)
        # Redis exists returns the number of keys that exist
        return cast(int, exists_result) > 0

    def keys(self) -> List[str]:
        """Get all keys in the channel."""
        pattern = f"{self.key_prefix}*"
        redis_keys = self.redis_client.keys(pattern)
        # Cast to list of strings for type safety
        redis_keys_list = cast(List[str], redis_keys)

        # Remove the prefix from each key
        prefix_len = len(self.key_prefix)
        result = []
        for key in redis_keys_list:
            # Convert to string and remove prefix
            if key.startswith(self.key_prefix):
                result.append(key[prefix_len:])
        return result

    def clear(self) -> None:
        """Clear all data from the channel."""
        pattern = f"{self.key_prefix}*"
        keys = self.redis_client.keys(pattern)
        # Cast to list for type safety
        keys_list = cast(List[Union[str, bytes]], keys)

        if keys_list:
            # Convert keys to ensure they're strings
            key_strs = [key if isinstance(key, str) else str(key) for key in keys_list]
            self.redis_client.delete(*key_strs)

    def ping(self) -> bool:
        """Check if Redis connection is alive."""
        try:
            return bool(self.redis_client.ping())
        except Exception:
            return False

    def close(self) -> None:
        """Close the Redis connection."""
        if hasattr(self.redis_client, 'close'):
            self.redis_client.close()
