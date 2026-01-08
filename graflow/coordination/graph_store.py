"""Content-Addressable Graph Storage on Redis."""

from __future__ import annotations

import hashlib
import logging
import zlib
from typing import TYPE_CHECKING

from cachetools import LRUCache

from graflow.core.graph import TaskGraph
from graflow.core.serialization import dumps, loads

if TYPE_CHECKING:
    from redis import Redis


class GraphStore:
    """Content-Addressable Graph Storage on Redis.

    Responsible for saving and loading graph snapshots.
    Integrates serialization, compression, and Redis operations.
    """

    DEFAULT_TTL = 86400  # 24 hours
    DEFAULT_CACHE_SIZE = 100  # Max number of graphs in LRU cache

    def __init__(
        self,
        redis_client: Redis,
        key_prefix: str,
        ttl: int = DEFAULT_TTL,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ):
        """Initialize GraphStore.

        Args:
            redis_client: Redis connection
            key_prefix: Redis key prefix (e.g., "app_name:workflows")
            ttl: Time-to-live in seconds (default: 24 hours)
            cache_size: LRU cache max size (default: 100 graphs)
        """
        self.redis = self._ensure_binary_client(redis_client)
        self.key_prefix = key_prefix
        self.ttl = ttl
        # LRU Cache: Prevent memory leaks in long-running workers
        self._local_cache: LRUCache[str, TaskGraph] = LRUCache(maxsize=cache_size)

    @staticmethod
    def _ensure_binary_client(redis_client: Redis) -> Redis:
        """
        Ensure the Redis client does not decode responses (required for binary graph blobs).

        Args:
            redis_client: Redis client (may be configured with decode_responses=True)

        Returns:
            A binary-safe Redis client. Returns the original client when already binary or
            when connection pool details are unavailable (e.g., during tests with mocks).
        """
        try:
            kwargs = dict(redis_client.connection_pool.connection_kwargs)
        except AttributeError:
            # Mock client or testing scenario - assume it's configured correctly
            return redis_client
        except Exception as e:
            # Unexpected error - log warning
            logging.warning(
                f"Could not inspect Redis client configuration: {e}. "
                f"Assuming binary mode is correct. If you encounter decode errors, "
                f"ensure decode_responses=False"
            )
            return redis_client

        if kwargs.get("decode_responses"):
            kwargs["decode_responses"] = False
            from redis import Redis

            return Redis(**kwargs)

        return redis_client

    def save(self, graph: TaskGraph) -> str:
        """Save graph and return its content-addressable hash.

        Same graph content results in same hash, preventing duplication.
        Compressed before storage to save bandwidth and memory.

        Args:
            graph: TaskGraph instance to save

        Returns:
            graph_hash: SHA256 hash of the graph (hex string)
        """
        # Serialize and calculate hash (before compression)
        graph_bytes = dumps(graph)
        graph_hash = hashlib.sha256(graph_bytes).hexdigest()

        # Check local cache first
        if graph_hash in self._local_cache:
            return graph_hash

        # Compress for storage efficiency
        compressed = zlib.compress(graph_bytes, level=6)

        # Save to Redis (atomic operation with SET NX EX)
        graph_key = f"{self.key_prefix}:graph:{graph_hash}"
        # SET NX EX: Set only if not exists (atomic, 1 round trip)
        self.redis.set(graph_key, compressed, nx=True, ex=self.ttl)

        # Update local cache
        self._local_cache[graph_hash] = graph

        return graph_hash

    def load(self, graph_hash: str) -> TaskGraph:
        """Restore graph from hash.

        Sliding TTL: Extends expire on every access to support long-running workflows.

        Args:
            graph_hash: SHA256 hash of the graph

        Returns:
            TaskGraph instance

        Raises:
            ValueError: If graph not found in Redis (TTL expired or never uploaded)
        """
        # Check local cache first (LRU)
        if graph_hash in self._local_cache:
            return self._local_cache[graph_hash]

        # Load from Redis
        graph_key = f"{self.key_prefix}:graph:{graph_hash}"
        compressed = self.redis.getex(graph_key, ex=self.ttl)

        if not compressed:
            raise ValueError(
                f"Graph snapshot not found: {graph_hash}\n"
                f"Key: {graph_key}\n"
                f"This may indicate:\n"
                f"  - Graph TTL expiration (current TTL: {self.ttl}s)\n"
                f"  - Missing graph upload (Lazy Upload not triggered)\n"
                f"  - Redis eviction (memory pressure)"
            )

        # Decompress and deserialize
        graph_bytes = zlib.decompress(compressed)  # type: ignore
        graph = loads(graph_bytes)

        # Update local cache (LRU)
        self._local_cache[graph_hash] = graph

        return graph
