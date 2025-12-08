"""Unit tests for RedisChannel SCAN-based keys() and clear() methods."""

import pytest

from graflow.channels.redis_channel import RedisChannel
from graflow.utils.redis_utils import create_redis_client, extract_redis_config


@pytest.mark.integration
class TestRedisChannelScanMethods:
    """Test RedisChannel SCAN-based methods (requires Redis)."""

    @pytest.fixture
    def redis_channel(self, clean_redis):
        """Create a Redis channel for testing."""
        redis_config = extract_redis_config(clean_redis)
        redis_config["decode_responses"] = True
        client = create_redis_client(redis_config)
        channel = RedisChannel("test_scan", redis_client=client)
        yield channel
        channel.clear()

    def test_keys_empty_channel(self, redis_channel):
        """Test keys() on empty channel returns empty list."""
        keys = redis_channel.keys()
        assert keys == []

    def test_keys_single_key(self, redis_channel):
        """Test keys() with a single key."""
        redis_channel.set("key1", "value1")

        keys = redis_channel.keys()
        assert len(keys) == 1
        assert "key1" in keys

    def test_keys_multiple_keys(self, redis_channel):
        """Test keys() with multiple keys."""
        # Add 10 keys
        for i in range(10):
            redis_channel.set(f"key_{i}", f"value_{i}")

        keys = redis_channel.keys()
        assert len(keys) == 10
        for i in range(10):
            assert f"key_{i}" in keys

    def test_keys_removes_prefix(self, redis_channel):
        """Test that keys() correctly removes the key prefix."""
        redis_channel.set("mykey", "myvalue")

        keys = redis_channel.keys()
        assert "mykey" in keys
        # Should not include the prefix
        assert not any("graflow:channel:" in key for key in keys)

    def test_keys_large_dataset(self, redis_channel):
        """Test keys() with large dataset (tests SCAN iteration)."""
        # Add 500 keys to test SCAN iteration beyond single batch
        num_keys = 500
        for i in range(num_keys):
            redis_channel.set(f"large_key_{i}", f"value_{i}")

        keys = redis_channel.keys()
        assert len(keys) == num_keys
        for i in range(num_keys):
            assert f"large_key_{i}" in keys

    def test_keys_handles_bytes_and_strings(self, redis_channel):
        """Test that keys() handles both bytes and string responses from Redis."""
        redis_channel.set("string_key", "value")
        redis_channel.set("another_key", "value2")

        keys = redis_channel.keys()
        assert len(keys) == 2
        assert all(isinstance(key, str) for key in keys)

    def test_clear_empty_channel(self, redis_channel):
        """Test clear() on empty channel does not raise error."""
        redis_channel.clear()  # Should not raise

        keys = redis_channel.keys()
        assert keys == []

    def test_clear_single_key(self, redis_channel):
        """Test clear() removes a single key."""
        redis_channel.set("key1", "value1")
        assert redis_channel.exists("key1")

        redis_channel.clear()

        assert not redis_channel.exists("key1")
        assert redis_channel.keys() == []

    def test_clear_multiple_keys(self, redis_channel):
        """Test clear() removes multiple keys."""
        # Add 20 keys
        for i in range(20):
            redis_channel.set(f"key_{i}", f"value_{i}")

        assert len(redis_channel.keys()) == 20

        redis_channel.clear()

        assert redis_channel.keys() == []
        for i in range(20):
            assert not redis_channel.exists(f"key_{i}")

    def test_clear_large_dataset_batching(self, redis_channel):
        """Test clear() with large dataset to verify batch deletion."""
        # Add 2500 keys to test batch deletion (batch_size is 1000)
        num_keys = 2500
        for i in range(num_keys):
            redis_channel.set(f"batch_key_{i}", f"value_{i}")

        assert len(redis_channel.keys()) == num_keys

        redis_channel.clear()

        assert redis_channel.keys() == []
        # Verify all keys are deleted
        for i in range(num_keys):
            assert not redis_channel.exists(f"batch_key_{i}")

    def test_clear_only_removes_channel_keys(self, clean_redis):
        """Test that clear() only removes keys for this channel, not others."""
        # Create two channels
        redis_config = extract_redis_config(clean_redis)
        redis_config["decode_responses"] = True
        client = create_redis_client(redis_config)

        channel1 = RedisChannel("channel1", redis_client=client)
        channel2 = RedisChannel("channel2", redis_client=client)

        # Add keys to both channels
        channel1.set("key1", "value1")
        channel1.set("key2", "value2")
        channel2.set("key1", "value1")
        channel2.set("key2", "value2")

        # Clear channel1
        channel1.clear()

        # channel1 should be empty
        assert channel1.keys() == []
        assert not channel1.exists("key1")
        assert not channel1.exists("key2")

        # channel2 should still have its keys
        assert len(channel2.keys()) == 2
        assert channel2.exists("key1")
        assert channel2.exists("key2")

        # Cleanup
        channel2.clear()

    def test_keys_and_clear_integration(self, redis_channel):
        """Test keys() and clear() work together correctly."""
        # Start with empty channel
        assert redis_channel.keys() == []

        # Add some keys
        for i in range(50):
            redis_channel.set(f"test_{i}", i)

        # Verify keys are present
        keys = redis_channel.keys()
        assert len(keys) == 50

        # Clear all keys
        redis_channel.clear()

        # Verify all keys are removed
        assert redis_channel.keys() == []

        # Add new keys to verify channel still works
        redis_channel.set("new_key", "new_value")
        assert redis_channel.keys() == ["new_key"]

    def test_scan_does_not_block_redis(self, redis_channel):
        """Test that SCAN-based operations don't block Redis (smoke test)."""
        # This is a smoke test to ensure we're using SCAN correctly
        # Add many keys
        num_keys = 1000
        for i in range(num_keys):
            redis_channel.set(f"key_{i}", f"value_{i}")

        # These operations should complete without timeout/blocking
        keys = redis_channel.keys()
        assert len(keys) == num_keys

        redis_channel.clear()
        assert redis_channel.keys() == []
