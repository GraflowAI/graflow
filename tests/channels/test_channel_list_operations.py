"""Unit tests for Channel list operations (append/prepend)."""

import pytest

from graflow.channels.memory_channel import MemoryChannel
from graflow.channels.redis_channel import RedisChannel
from graflow.utils.redis_utils import create_redis_client, extract_redis_config


class TestMemoryChannelListOperations:
    """Test MemoryChannel list operations."""

    def test_append_to_new_list(self):
        """Test appending to a new list creates the list."""
        channel = MemoryChannel("test")

        length = channel.append("mylist", "first")
        assert length == 1

        data = channel.get("mylist")
        assert data == ["first"]

    def test_append_multiple_values(self):
        """Test appending multiple values to a list."""
        channel = MemoryChannel("test")

        channel.append("mylist", "first")
        channel.append("mylist", "second")
        length = channel.append("mylist", "third")

        assert length == 3

        data = channel.get("mylist")
        assert data == ["first", "second", "third"]

    def test_prepend_to_new_list(self):
        """Test prepending to a new list creates the list."""
        channel = MemoryChannel("test")

        length = channel.prepend("mylist", "first")
        assert length == 1

        data = channel.get("mylist")
        assert data == ["first"]

    def test_prepend_multiple_values(self):
        """Test prepending multiple values to a list."""
        channel = MemoryChannel("test")

        channel.prepend("mylist", "third")
        channel.prepend("mylist", "second")
        length = channel.prepend("mylist", "first")

        assert length == 3

        data = channel.get("mylist")
        assert data == ["first", "second", "third"]

    def test_mixed_append_prepend(self):
        """Test mixing append and prepend operations."""
        channel = MemoryChannel("test")

        channel.append("mylist", "middle")
        channel.prepend("mylist", "first")
        channel.append("mylist", "last")

        data = channel.get("mylist")
        assert data == ["first", "middle", "last"]

    def test_append_with_ttl(self):
        """Test append with TTL."""
        import time

        channel = MemoryChannel("test")

        channel.append("mylist", "value", ttl=1)

        # Should exist immediately
        assert channel.exists("mylist")
        data = channel.get("mylist")
        assert data == ["value"]

        # Should expire after TTL
        time.sleep(1.1)
        assert not channel.exists("mylist")

    def test_prepend_with_ttl(self):
        """Test prepend with TTL."""
        import time

        channel = MemoryChannel("test")

        channel.prepend("mylist", "value", ttl=1)

        # Should exist immediately
        assert channel.exists("mylist")

        # Should expire after TTL
        time.sleep(1.1)
        assert not channel.exists("mylist")

    def test_append_to_non_list_raises_error(self):
        """Test appending to a non-list key raises TypeError."""
        channel = MemoryChannel("test")

        # Set a non-list value
        channel.set("notalist", "string_value")

        # Appending should raise TypeError
        with pytest.raises(TypeError, match="not a list"):
            channel.append("notalist", "value")

    def test_prepend_to_non_list_raises_error(self):
        """Test prepending to a non-list key raises TypeError."""
        channel = MemoryChannel("test")

        # Set a non-list value
        channel.set("notalist", {"key": "value"})

        # Prepending should raise TypeError
        with pytest.raises(TypeError, match="not a list"):
            channel.prepend("notalist", "value")


@pytest.mark.integration
class TestRedisChannelListOperations:
    """Test RedisChannel list operations (requires Redis)."""

    @pytest.fixture
    def redis_channel(self, clean_redis):
        """Create a Redis channel for testing."""
        redis_config = extract_redis_config(clean_redis)
        redis_config["decode_responses"] = True
        client = create_redis_client(redis_config)
        channel = RedisChannel("test", redis_client=client)
        yield channel
        channel.clear()

    def test_append_to_new_list(self, redis_channel):
        """Test appending to a new list creates the list."""
        length = redis_channel.append("mylist", "first")
        assert length == 1

        # Verify with direct Redis access
        import redis

        client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        result = client.lrange("graflow:channel:test:mylist", 0, -1)
        assert isinstance(result, list)

        # Parse JSON values
        import json

        parsed = [json.loads(v) for v in result]
        assert parsed == ["first"]

    def test_append_multiple_values(self, redis_channel):
        """Test appending multiple values to a list."""
        redis_channel.append("mylist", "first")
        redis_channel.append("mylist", "second")
        length = redis_channel.append("mylist", "third")

        assert length == 3

        # Verify with direct Redis access
        import json

        import redis

        client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        result = client.lrange("graflow:channel:test:mylist", 0, -1)
        assert isinstance(result, list)
        parsed = [json.loads(v) for v in result]
        assert parsed == ["first", "second", "third"]

    def test_prepend_to_new_list(self, redis_channel):
        """Test prepending to a new list creates the list."""
        length = redis_channel.prepend("mylist", "first")
        assert length == 1

        # Verify with direct Redis access
        import json

        import redis

        client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        result = client.lrange("graflow:channel:test:mylist", 0, -1)
        assert isinstance(result, list)
        parsed = [json.loads(v) for v in result]
        assert parsed == ["first"]

    def test_prepend_multiple_values(self, redis_channel):
        """Test prepending multiple values to a list."""
        redis_channel.prepend("mylist", "third")
        redis_channel.prepend("mylist", "second")
        length = redis_channel.prepend("mylist", "first")

        assert length == 3

        # Verify with direct Redis access
        import json

        import redis

        client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        result = client.lrange("graflow:channel:test:mylist", 0, -1)
        assert isinstance(result, list)
        parsed = [json.loads(v) for v in result]
        assert parsed == ["first", "second", "third"]

    def test_mixed_append_prepend(self, redis_channel):
        """Test mixing append and prepend operations."""
        redis_channel.append("mylist", "middle")
        redis_channel.prepend("mylist", "first")
        redis_channel.append("mylist", "last")

        # Verify with direct Redis access
        import json

        import redis

        client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        result = client.lrange("graflow:channel:test:mylist", 0, -1)
        assert isinstance(result, list)
        parsed = [json.loads(v) for v in result]
        assert parsed == ["first", "middle", "last"]

    def test_append_with_ttl(self, redis_channel):
        """Test append with TTL."""
        import time

        redis_channel.append("mylist", "value", ttl=1)

        # Should exist immediately
        assert redis_channel.exists("mylist")

        # Should expire after TTL
        time.sleep(1.1)
        assert not redis_channel.exists("mylist")

    def test_prepend_with_ttl(self, redis_channel):
        """Test prepend with TTL."""
        import time

        redis_channel.prepend("mylist", "value", ttl=1)

        # Should exist immediately
        assert redis_channel.exists("mylist")

        # Should expire after TTL
        time.sleep(1.1)
        assert not redis_channel.exists("mylist")

    def test_append_complex_types(self, redis_channel):
        """Test appending complex types (dicts, lists, etc)."""
        redis_channel.append("mylist", {"key": "value"})
        redis_channel.append("mylist", [1, 2, 3])
        redis_channel.append("mylist", 42)

        # Verify with direct Redis access
        import json

        import redis

        client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        result = client.lrange("graflow:channel:test:mylist", 0, -1)
        assert isinstance(result, list)
        parsed = [json.loads(v) for v in result]
        assert parsed == [{"key": "value"}, [1, 2, 3], 42]
