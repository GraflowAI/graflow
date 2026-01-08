"""Redis-based channel implementation for inter-task communication."""

from __future__ import annotations

import json
from typing import Any, List, Optional, cast

from graflow.channels.base import Channel

try:
    import redis
    from redis import Redis
except ImportError:
    redis = None


class RedisChannel(Channel):
    """Redis-based channel implementation for inter-task communication."""

    def __init__(
        self,
        name: str,
        redis_client: Optional[Redis] = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        **kwargs,
    ):
        """Initialize Redis channel.

        Args:
            name: Channel name
            redis_client: Optional Redis client instance. If not provided, creates new one with host/port/db
            host: Redis host (used only if redis_client is None)
            port: Redis port (used only if redis_client is None)
            db: Redis database (used only if redis_client is None)
            **kwargs: Additional arguments passed to Redis constructor (used only if redis_client is None)
        """
        super().__init__(name)
        if redis is None:
            raise ImportError("redis package is required for RedisChannel")

        # Store connection parameters for serialization
        self._host = host
        self._port = port
        self._db = db
        self._kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            in {
                "password",
                "socket_timeout",
                "socket_connect_timeout",
                "socket_keepalive",
                "socket_keepalive_options",
                "connection_pool",
                "unix_socket_path",
                "encoding",
                "encoding_errors",
                "charset",
                "errors",
                "decode_responses",
                "retry_on_timeout",
                "ssl",
                "ssl_keyfile",
                "ssl_certfile",
                "ssl_cert_reqs",
                "ssl_ca_certs",
                "ssl_check_hostname",
                "max_connections",
                "retry",
                "health_check_interval",
            }
        }

        if redis_client is not None:
            self.redis_client = redis_client
        else:
            # Prefer caller-provided decode_responses, defaulting to True otherwise.
            decode_responses = self._kwargs.pop("decode_responses", True)
            self.redis_client: Redis = redis.Redis(
                host=host, port=port, db=db, decode_responses=decode_responses, **self._kwargs
            )

        self.key_prefix = f"graflow:channel:{name}:"

    def _get_key(self, key: str) -> str:
        """Get the full Redis key with prefix."""
        return f"{self.key_prefix}{key}"

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store data in the channel."""
        redis_key = self._get_key(key)
        serialized_value = json.dumps(value, default=str)

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
        """Get all keys in the channel using SCAN (non-blocking)."""
        pattern = f"{self.key_prefix}*"
        matched_keys = []

        # Cursor-based scan (non-blocking)
        for key in self.redis_client.scan_iter(match=pattern, count=100):
            # Remove key_prefix from the key
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            matched_keys.append(key_str.replace(f"{self.key_prefix}", "", 1))

        return matched_keys

    def clear(self) -> None:
        """Clear all data from the channel using SCAN (paginated)."""
        pattern = f"{self.key_prefix}*"
        batch_size = 1000
        keys_to_delete = []

        # Batch scan and delete
        for key in self.redis_client.scan_iter(match=pattern, count=100):
            keys_to_delete.append(key)
            if len(keys_to_delete) >= batch_size:
                if keys_to_delete:
                    self.redis_client.delete(*keys_to_delete)
                keys_to_delete = []

        # Delete remaining keys
        if keys_to_delete:
            self.redis_client.delete(*keys_to_delete)

    def ping(self) -> bool:
        """Check if Redis connection is alive."""
        try:
            return bool(self.redis_client.ping())
        except Exception:
            return False

    def close(self) -> None:
        """Close the Redis connection."""
        if hasattr(self.redis_client, "close"):
            self.redis_client.close()

    def append(self, key: str, value: Any, ttl: Optional[int] = None) -> int:
        """Append value to a list stored at key using Redis RPUSH.

        Args:
            key: The key identifying the list
            value: Value to append to the list
            ttl: Optional time-to-live in seconds for the key

        Returns:
            Length of the list after append
        """
        redis_key = self._get_key(key)
        serialized_value = json.dumps(value, default=str)

        # Use RPUSH to append to the end of the list
        length = self.redis_client.rpush(redis_key, serialized_value)

        # Set TTL if specified
        if ttl is not None:
            self.redis_client.expire(redis_key, ttl)

        return cast(int, length)

    def prepend(self, key: str, value: Any, ttl: Optional[int] = None) -> int:
        """Prepend value to the head of a list stored at key using Redis LPUSH.

        Args:
            key: The key identifying the list
            value: Value to prepend to the list
            ttl: Optional time-to-live in seconds for the key

        Returns:
            Length of the list after prepend
        """
        redis_key = self._get_key(key)
        serialized_value = json.dumps(value, default=str)

        # Use LPUSH to prepend to the head of the list
        length = self.redis_client.lpush(redis_key, serialized_value)

        # Set TTL if specified
        if ttl is not None:
            self.redis_client.expire(redis_key, ttl)

        return cast(int, length)

    def __getstate__(self):
        """Support for pickle serialization."""
        state = self.__dict__.copy()
        # Remove the unpicklable Redis client
        del state["redis_client"]
        return state

    def __setstate__(self, state):
        """Support for pickle deserialization."""
        self.__dict__.update(state)
        # Recreate the Redis client
        assert redis is not None, "redis package is required for RedisChannel"
        self.redis_client = redis.Redis(
            host=self._host, port=self._port, db=self._db, decode_responses=True, **self._kwargs
        )
