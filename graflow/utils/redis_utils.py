"""Redis client utilities for serialization support.

This module provides utilities to handle Redis client serialization issues.
Redis client objects contain thread locks and cannot be pickled, so we extract
connection parameters and recreate clients as needed.
"""

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from redis import Redis

try:
    from redis import Redis as _Redis  # Runtime import

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    _Redis = None  # type: ignore[assignment,misc]


# Redis connection parameters with their defaults
_REDIS_PARAMS = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "decode_responses": False,
    "password": None,
    "username": None,
    "socket_timeout": None,
    "ssl": None,
}


def extract_redis_config(redis_client: "Redis") -> Dict[str, Any]:
    """Extract connection parameters from Redis client.

    Args:
        redis_client: Redis client instance

    Returns:
        Dictionary with connection parameters (host, port, db, password, etc.)

    Raises:
        ImportError: If redis package is not available
        AttributeError: If redis_client is not a valid Redis instance

    Example:
        >>> import redis
        >>> client = redis.Redis(host='localhost', port=6379, db=1, password='secret')
        >>> config = extract_redis_config(client)
        >>> config['host']
        'localhost'
        >>> config['port']
        6379
        >>> config['password']
        'secret'
    """
    if not REDIS_AVAILABLE:
        raise ImportError("Redis package is not available")

    conn_kwargs = redis_client.connection_pool.connection_kwargs

    config = {}
    for key, default_value in _REDIS_PARAMS.items():
        value = conn_kwargs.get(key, default_value)
        # Only include non-None values or required params
        if value is not None or key in {"host", "port", "db", "decode_responses"}:
            config[key] = value

    return config


def create_redis_client(config: Dict[str, Any]) -> "Redis":
    """Create Redis client from connection parameters.

    If config already contains a 'redis_client' key, validates and returns it
    instead of creating a new client. This avoids unnecessary client recreation.

    Args:
        config: Dictionary with connection parameters (host, port, db, password, etc.)
            or an existing 'redis_client' instance

    Returns:
        Redis client instance (existing or newly created)

    Raises:
        ImportError: If redis package is not available
        AssertionError: If redis_client exists but is not a valid Redis instance

    Example:
        >>> config = {'host': 'localhost', 'port': 6379, 'db': 0, 'password': 'secret'}
        >>> client = create_redis_client(config)
        >>> client.ping()
        True

        >>> # Reuse existing client
        >>> config_with_client = {'redis_client': existing_client}
        >>> client = create_redis_client(config_with_client)
        >>> client is existing_client
        True
    """
    if not REDIS_AVAILABLE or _Redis is None:
        raise ImportError("Redis package is not available")

    # If redis_client already exists, validate and return it
    if "redis_client" in config:
        redis_client = config["redis_client"]

        assert isinstance(redis_client, _Redis), f"Expected Redis instance, got {type(redis_client).__name__}"

        return redis_client

    # Build Redis constructor kwargs from config
    redis_kwargs = {}
    for key, default_value in _REDIS_PARAMS.items():
        value = config.get(key, default_value)
        # Only include non-None values
        if value is not None:
            redis_kwargs[key] = value

    return _Redis(**redis_kwargs)


def ensure_redis_connection_params(config: Dict[str, Any]) -> None:
    """Ensure redis_client in config has corresponding connection parameters.

    If 'redis_client' exists in config, extracts connection params and adds them
    to the config dict (only if not already present). The redis_client key itself
    remains in the dict and should be removed later during serialization.

    This ensures connection parameters are available for recreating the client
    while still allowing redis_client to be used at runtime.

    Args:
        config: Configuration dict (may contain 'redis_client')
            Modified in-place to include connection parameters.

    Raises:
        ImportError: If redis package is not available
        AssertionError: If redis_client is not a valid Redis instance
        ValueError: If redis_client extraction fails (fail-fast)

    Example:
        >>> import redis
        >>> client = redis.Redis(host='localhost', port=6379)
        >>> config = {'redis_client': client, 'key_prefix': 'graflow'}
        >>> ensure_redis_connection_params(config)
        >>> config['host']
        'localhost'
        >>> config['key_prefix']
        'graflow'
        >>> 'redis_client' in config  # Still present until serialization
        True
    """
    if "redis_client" not in config:
        return

    redis_client = config["redis_client"]

    # Validate redis_client type
    if not REDIS_AVAILABLE or _Redis is None:
        raise ImportError("Redis package is not available")

    assert isinstance(redis_client, _Redis), f"Expected Redis instance, got {type(redis_client).__name__}"

    try:
        redis_config = extract_redis_config(redis_client)
        # Update config with connection params (only if not already present)
        for key, value in redis_config.items():
            if key not in config:
                config[key] = value
    except Exception as e:
        # Fail fast: Don't silently use defaults
        # Silent defaults could connect to wrong Redis instance
        raise ValueError(f"Failed to extract Redis connection parameters from redis_client: {e}") from e
