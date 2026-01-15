from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from cachetools import TLRUCache as Cachetools_TLRUCache

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CachedValue(Generic[V]):
    # Holds the user value along with per-entry TTL (seconds).
    # ttl == math.inf means "never expire".
    value: V
    ttl: float


class TLRUCache(Generic[K, V]):
    """
    Per-entry TTL + LRU cache built on top of cachetools.TLRUCache.

    Semantics:
    - Expiration deadline is computed per entry as: now + cached.ttl
    - ttl_seconds == 0 means "never expire"
    - set(..., ttl_seconds=0) stores an infinite TTL
    - get() does NOT refresh TTL by default (non-sliding)
    - get(..., ttl_seconds=...) explicitly refreshes TTL
    - touch(..., ttl_seconds=...) updates TTL and recomputes expiration
    """

    def __init__(self, *, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")

        # Compute absolute expiration time per entry.
        def ttu(_key: K, cached: CachedValue[V], now: float) -> float:
            return now + cached.ttl

        # Use explicit alias to avoid name collision with this class.
        self._cache: Cachetools_TLRUCache = Cachetools_TLRUCache(maxsize=maxsize, ttu=ttu)

    def set(self, key: K, value: V, *, ttl_seconds: float = 0) -> None:
        """
        Set a value with per-entry TTL.

        ttl_seconds:
          - 0   : never expire (default)
          - >0  : expire after ttl_seconds
        """
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")

        ttl = math.inf if ttl_seconds == 0 else float(ttl_seconds)
        self._cache[key] = CachedValue(value=value, ttl=ttl)

    def get(
        self,
        key: K,
        default: Optional[V] = None,
        *,
        ttl_seconds: Optional[float] = None,
    ) -> Optional[V]:
        """
        Get a value.

        - ttl_seconds is None: no TTL refresh
        - ttl_seconds is provided: refresh TTL via touch()
        """
        cached = self._cache.get(key)
        if cached is None:
            return default

        if ttl_seconds is not None:
            self.touch(key, ttl_seconds=ttl_seconds)

        return cached.value

    def touch(self, key: K, *, ttl_seconds: float) -> bool:
        """
        Update TTL for an existing key.

        ttl_seconds:
          - 0   : never expire
          - >0  : new TTL in seconds
        """
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be >= 0")

        cached = self._cache.get(key)
        if cached is None:
            return False

        cached.ttl = math.inf if ttl_seconds == 0 else float(ttl_seconds)
        # Reinsert to force expiration recomputation
        self._cache[key] = cached
        return True

    def delete(self, key: K) -> bool:
        sentinel = object()
        removed = self._cache.pop(key, sentinel)
        return removed is not sentinel

    def contains(self, key: K) -> bool:
        return self._cache.get(key) is not None

    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
