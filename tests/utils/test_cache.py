"""Unit tests for TLRUCache implementation.

Test Coverage:
- TestCachedValue: Dataclass creation, infinite TTL storage
- TestTLRUCacheInit: Initialization, maxsize validation (positive required)
- TestTLRUCacheSet: Set operations, TTL configuration, LRU eviction on overflow
- TestTLRUCacheGet: Get operations, default values, TTL refresh behavior
- TestTLRUCacheExpiration: TTL expiration, per-entry TTL, zero TTL = never expires
- TestTLRUCacheTouch: TTL update, make entry permanent, nonexistent key handling
- TestTLRUCacheDelete: Delete operations, return value semantics
- TestTLRUCacheContains: Existence check, expired entry detection
- TestTLRUCacheSizeAndClear: Size retrieval, cache clearing
- TestTLRUCacheIntegration: Complex workflows, type safety with generics

Key Behaviors Verified:
- ttl_seconds=0 means "never expire" (stored as math.inf)
- ttl_seconds>0 means "expire after N seconds"
- get() does NOT refresh TTL by default (non-sliding window)
- get(ttl_seconds=...) explicitly refreshes TTL
- touch() updates TTL and recomputes expiration deadline
- LRU eviction occurs when cache exceeds maxsize
"""

from __future__ import annotations

import time

import pytest

from graflow.utils.cache import CachedValue, TLRUCache


class TestCachedValue:
    """Tests for CachedValue dataclass."""

    def test_cached_value_creation(self):
        """Test CachedValue holds value and ttl."""
        cached = CachedValue(value="test", ttl=60.0)
        assert cached.value == "test"
        assert cached.ttl == 60.0

    def test_cached_value_with_infinite_ttl(self):
        """Test CachedValue with infinite TTL."""
        import math

        cached = CachedValue(value=123, ttl=math.inf)
        assert cached.value == 123
        assert cached.ttl == math.inf


class TestTLRUCacheInit:
    """Tests for TLRUCache initialization."""

    def test_init_with_valid_maxsize(self):
        """Test initialization with valid maxsize."""
        cache: TLRUCache[str, int] = TLRUCache(maxsize=100)
        assert cache.size() == 0

    def test_init_with_zero_maxsize_raises(self):
        """Test initialization with zero maxsize raises ValueError."""
        with pytest.raises(ValueError, match="maxsize must be positive"):
            TLRUCache(maxsize=0)

    def test_init_with_negative_maxsize_raises(self):
        """Test initialization with negative maxsize raises ValueError."""
        with pytest.raises(ValueError, match="maxsize must be positive"):
            TLRUCache(maxsize=-1)


class TestTLRUCacheSet:
    """Tests for TLRUCache.set()."""

    def test_set_with_default_ttl(self):
        """Test set with default TTL (never expires)."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value")

        assert cache.get("key") == "value"
        assert cache.size() == 1

    def test_set_with_explicit_ttl(self):
        """Test set with explicit TTL."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value", ttl_seconds=60)

        assert cache.get("key") == "value"

    def test_set_with_zero_ttl_never_expires(self):
        """Test set with ttl_seconds=0 means never expire."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value", ttl_seconds=0)

        assert cache.get("key") == "value"

    def test_set_with_negative_ttl_raises(self):
        """Test set with negative TTL raises ValueError."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)

        with pytest.raises(ValueError, match="ttl_seconds must be >= 0"):
            cache.set("key", "value", ttl_seconds=-1)

    def test_set_overwrites_existing_key(self):
        """Test set overwrites existing key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value1")
        cache.set("key", "value2")

        assert cache.get("key") == "value2"
        assert cache.size() == 1

    def test_set_evicts_lru_when_full(self):
        """Test LRU eviction when cache is full."""
        cache: TLRUCache[str, int] = TLRUCache(maxsize=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        # Access "a" to make it recently used
        cache.get("a")

        # Add new item - should evict "b" (least recently used)
        cache.set("d", 4)

        assert cache.get("a") == 1  # Still present (recently used)
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == 3
        assert cache.get("d") == 4


class TestTLRUCacheGet:
    """Tests for TLRUCache.get()."""

    def test_get_existing_key(self):
        """Test get returns value for existing key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value")

        assert cache.get("key") == "value"

    def test_get_nonexistent_key_returns_none(self):
        """Test get returns None for nonexistent key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)

        assert cache.get("nonexistent") is None

    def test_get_nonexistent_key_returns_default(self):
        """Test get returns default for nonexistent key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)

        assert cache.get("nonexistent", "default") == "default"

    def test_get_does_not_refresh_ttl_by_default(self):
        """Test get does not refresh TTL by default (non-sliding)."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value", ttl_seconds=0.1)

        # Access without refreshing TTL
        cache.get("key")

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        assert cache.get("key") is None

    def test_get_with_ttl_seconds_refreshes_ttl(self):
        """Test get with ttl_seconds refreshes TTL."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value", ttl_seconds=0.1)

        # Wait a bit
        time.sleep(0.05)

        # Refresh TTL
        cache.get("key", ttl_seconds=1.0)

        # Wait past original expiration
        time.sleep(0.1)

        # Should still be present (TTL was refreshed)
        assert cache.get("key") == "value"


class TestTLRUCacheExpiration:
    """Tests for TTL expiration behavior."""

    def test_entry_expires_after_ttl(self):
        """Test entry expires after TTL."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value", ttl_seconds=0.1)

        # Initially present
        assert cache.get("key") == "value"

        # Wait for expiration
        time.sleep(0.15)

        # Should be expired
        assert cache.get("key") is None

    def test_zero_ttl_never_expires(self):
        """Test ttl_seconds=0 means never expire."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value", ttl_seconds=0)

        # Wait a bit
        time.sleep(0.1)

        # Should still be present
        assert cache.get("key") == "value"

    def test_different_entries_different_ttls(self):
        """Test different entries can have different TTLs."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("short", "value1", ttl_seconds=0.1)
        cache.set("long", "value2", ttl_seconds=1.0)
        cache.set("never", "value3", ttl_seconds=0)

        # Wait for short TTL to expire
        time.sleep(0.15)

        assert cache.get("short") is None  # Expired
        assert cache.get("long") == "value2"  # Still valid
        assert cache.get("never") == "value3"  # Never expires


class TestTLRUCacheTouch:
    """Tests for TLRUCache.touch()."""

    def test_touch_existing_key(self):
        """Test touch updates TTL for existing key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value", ttl_seconds=0.1)

        # Touch with new TTL
        result = cache.touch("key", ttl_seconds=1.0)

        assert result is True

        # Wait past original expiration
        time.sleep(0.15)

        # Should still be present
        assert cache.get("key") == "value"

    def test_touch_nonexistent_key(self):
        """Test touch returns False for nonexistent key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)

        result = cache.touch("nonexistent", ttl_seconds=60)

        assert result is False

    def test_touch_with_zero_ttl_makes_permanent(self):
        """Test touch with ttl_seconds=0 makes entry permanent."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value", ttl_seconds=0.1)

        # Make permanent
        cache.touch("key", ttl_seconds=0)

        # Wait past original expiration
        time.sleep(0.15)

        # Should still be present (now permanent)
        assert cache.get("key") == "value"

    def test_touch_with_negative_ttl_raises(self):
        """Test touch with negative TTL raises ValueError."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value")

        with pytest.raises(ValueError, match="ttl_seconds must be >= 0"):
            cache.touch("key", ttl_seconds=-1)


class TestTLRUCacheDelete:
    """Tests for TLRUCache.delete()."""

    def test_delete_existing_key(self):
        """Test delete removes existing key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value")

        result = cache.delete("key")

        assert result is True
        assert cache.get("key") is None
        assert cache.size() == 0

    def test_delete_nonexistent_key(self):
        """Test delete returns False for nonexistent key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)

        result = cache.delete("nonexistent")

        assert result is False


class TestTLRUCacheContains:
    """Tests for TLRUCache.contains()."""

    def test_contains_existing_key(self):
        """Test contains returns True for existing key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value")

        assert cache.contains("key") is True

    def test_contains_nonexistent_key(self):
        """Test contains returns False for nonexistent key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)

        assert cache.contains("nonexistent") is False

    def test_contains_expired_key(self):
        """Test contains returns False for expired key."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)
        cache.set("key", "value", ttl_seconds=0.1)

        # Wait for expiration
        time.sleep(0.15)

        assert cache.contains("key") is False


class TestTLRUCacheSizeAndClear:
    """Tests for TLRUCache.size() and clear()."""

    def test_size_empty_cache(self):
        """Test size returns 0 for empty cache."""
        cache: TLRUCache[str, str] = TLRUCache(maxsize=10)

        assert cache.size() == 0

    def test_size_with_entries(self):
        """Test size returns correct count."""
        cache: TLRUCache[str, int] = TLRUCache(maxsize=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        assert cache.size() == 3

    def test_clear_removes_all_entries(self):
        """Test clear removes all entries."""
        cache: TLRUCache[str, int] = TLRUCache(maxsize=10)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)

        cache.clear()

        assert cache.size() == 0
        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") is None


class TestTLRUCacheIntegration:
    """Integration tests for TLRUCache."""

    def test_complex_workflow(self):
        """Test complex workflow with set, get, touch, delete."""
        cache: TLRUCache[str, dict] = TLRUCache(maxsize=5)

        # Set entries with different TTLs
        cache.set("user:1", {"name": "Alice"}, ttl_seconds=0)  # Never expires
        cache.set("user:2", {"name": "Bob"}, ttl_seconds=0.2)
        cache.set("session:1", {"token": "abc"}, ttl_seconds=0.1)

        # Verify all present
        assert cache.size() == 3
        assert cache.get("user:1") == {"name": "Alice"}
        assert cache.get("user:2") == {"name": "Bob"}
        assert cache.get("session:1") == {"token": "abc"}

        # Touch session to extend TTL
        cache.touch("session:1", ttl_seconds=1.0)

        # Wait for user:2's original TTL
        time.sleep(0.25)

        # user:2 should be expired, but session:1 extended
        assert cache.get("user:1") == {"name": "Alice"}  # Never expires
        assert cache.get("user:2") is None  # Expired
        assert cache.get("session:1") == {"token": "abc"}  # Extended

        # Delete user:1
        assert cache.delete("user:1") is True
        assert cache.get("user:1") is None

    def test_type_safety_with_generics(self):
        """Test cache works correctly with different types."""
        # String keys, int values
        int_cache: TLRUCache[str, int] = TLRUCache(maxsize=10)
        int_cache.set("count", 42)
        assert int_cache.get("count") == 42

        # Int keys, string values
        str_cache: TLRUCache[int, str] = TLRUCache(maxsize=10)
        str_cache.set(1, "one")
        assert str_cache.get(1) == "one"

        # Tuple keys, list values
        complex_cache: TLRUCache[tuple, list] = TLRUCache(maxsize=10)
        complex_cache.set(("a", 1), [1, 2, 3])
        assert complex_cache.get(("a", 1)) == [1, 2, 3]
