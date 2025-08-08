"""Tests for typed channel functionality."""

import time
from typing import TypedDict

import pytest

from graflow.channels.memory import MemoryChannel
from graflow.channels.schemas import TaskProgressMessage, TaskResultMessage
from graflow.channels.typed import ChannelTypeRegistry, TypedChannel, _is_typed_dict, _validate_typed_dict


class TestMessage(TypedDict):
    """Test message type."""
    id: str
    value: int
    optional_field: str


class InvalidMessage:
    """Not a TypedDict."""
    def __init__(self):
        self.id = "test"


class TestTypedChannel:
    """Test TypedChannel functionality."""

    def test_create_typed_channel(self):
        """Test creating a typed channel."""
        memory_channel = MemoryChannel("test")
        typed_channel = TypedChannel(memory_channel, TestMessage)

        assert typed_channel.name == "test"
        assert typed_channel.message_type == TestMessage

    def test_invalid_message_type(self):
        """Test error when using non-TypedDict."""
        memory_channel = MemoryChannel("test")

        with pytest.raises(ValueError, match="message_type must be a TypedDict"):
            TypedChannel(memory_channel, InvalidMessage)

    def test_send_valid_message(self):
        """Test sending a valid message."""
        memory_channel = MemoryChannel("test")
        typed_channel = TypedChannel(memory_channel, TestMessage)

        message: TestMessage = {
            "id": "test123",
            "value": 42,
            "optional_field": "hello"
        }

        typed_channel.send("key1", message)
        assert typed_channel.exists("key1")

    def test_send_invalid_message(self):
        """Test error when sending invalid message."""
        memory_channel = MemoryChannel("test")
        typed_channel = TypedChannel(memory_channel, TestMessage)

        # Missing required field
        invalid_message = {"id": "test123"}

        with pytest.raises(TypeError, match="Message does not conform"):
            typed_channel.send("key1", invalid_message) # type: ignore

    def test_receive_valid_message(self):
        """Test receiving a valid message."""
        memory_channel = MemoryChannel("test")
        typed_channel = TypedChannel(memory_channel, TestMessage)

        message: TestMessage = {
            "id": "test123",
            "value": 42,
            "optional_field": "hello"
        }

        typed_channel.send("key1", message)
        received = typed_channel.receive("key1")
        assert received is not None

        assert received == message
        assert received["id"] == "test123"
        assert received["value"] == 42

    def test_receive_nonexistent_key(self):
        """Test receiving from nonexistent key."""
        memory_channel = MemoryChannel("test")
        typed_channel = TypedChannel(memory_channel, TestMessage)

        result = typed_channel.receive("nonexistent")
        assert result is None

    def test_receive_invalid_data(self):
        """Test receiving invalid data returns None."""
        memory_channel = MemoryChannel("test")
        typed_channel = TypedChannel(memory_channel, TestMessage)

        # Manually set invalid data in underlying channel
        memory_channel.set("key1", {"invalid": "data"})

        result = typed_channel.receive("key1")
        assert result is None

    def test_channel_operations(self):
        """Test basic channel operations."""
        memory_channel = MemoryChannel("test")
        typed_channel = TypedChannel(memory_channel, TestMessage)

        message: TestMessage = {
            "id": "test123",
            "value": 42,
            "optional_field": "hello"
        }

        # Send and check existence
        typed_channel.send("key1", message)
        assert typed_channel.exists("key1")

        # Check keys
        keys = typed_channel.keys()
        assert "key1" in keys

        # Delete
        result = typed_channel.delete("key1")
        assert result is True
        assert not typed_channel.exists("key1")

        # Clear
        typed_channel.send("key2", message)
        typed_channel.clear()
        assert len(typed_channel.keys()) == 0


class TestChannelTypeRegistry:
    """Test ChannelTypeRegistry functionality."""

    def test_register_and_get_type(self):
        """Test registering and retrieving message types."""
        ChannelTypeRegistry.register("test_msg", TestMessage)
        retrieved = ChannelTypeRegistry.get("test_msg")
        assert retrieved == TestMessage

    def test_register_invalid_type(self):
        """Test error when registering non-TypedDict."""
        with pytest.raises(ValueError, match="message_type must be a TypedDict"):
            ChannelTypeRegistry.register("invalid", InvalidMessage)

    def test_get_nonexistent_type(self):
        """Test getting nonexistent type returns None."""
        result = ChannelTypeRegistry.get("nonexistent")
        assert result is None

    def test_create_channel_from_registry(self):
        """Test creating typed channel from registry."""
        ChannelTypeRegistry.register("test_msg", TestMessage)
        memory_channel = MemoryChannel("test")

        typed_channel = ChannelTypeRegistry.create_channel(memory_channel, "test_msg")
        assert typed_channel.message_type == TestMessage

    def test_create_channel_nonexistent_type(self):
        """Test error when creating channel with nonexistent type."""
        memory_channel = MemoryChannel("test")

        with pytest.raises(ValueError, match="Message type 'nonexistent' not registered"):
            ChannelTypeRegistry.create_channel(memory_channel, "nonexistent")


class TestSchemas:
    """Test predefined message schemas."""

    def test_task_result_message(self):
        """Test TaskResultMessage schema."""
        memory_channel = MemoryChannel("test")
        typed_channel = TypedChannel(memory_channel, TaskResultMessage)

        message: TaskResultMessage = {
            "task_id": "task1",
            "result": {"processed": True},
            "timestamp": time.time(),
            "status": "completed"
        }

        typed_channel.send("result", message)
        received = typed_channel.receive("result")

        assert received is not None
        assert received["task_id"] == "task1"
        assert received["status"] == "completed"

    def test_task_progress_message(self):
        """Test TaskProgressMessage schema."""
        memory_channel = MemoryChannel("test")
        typed_channel = TypedChannel(memory_channel, TaskProgressMessage)

        message: TaskProgressMessage = {
            "task_id": "task1",
            "progress": 0.75,
            "message": "Processing...",
            "timestamp": time.time()
        }

        typed_channel.send("progress", message)
        received = typed_channel.receive("progress")

        assert received is not None
        assert received["progress"] == 0.75
        assert received["message"] == "Processing..."


class TestValidationHelpers:
    """Test validation helper functions."""

    def test_is_typed_dict(self):
        """Test TypedDict detection."""
        assert _is_typed_dict(TestMessage) is True
        assert _is_typed_dict(InvalidMessage) is False
        assert _is_typed_dict(dict) is False

    def test_validate_typed_dict_valid(self):
        """Test validation with valid data."""
        data = {
            "id": "test123",
            "value": 42,
            "optional_field": "hello"
        }
        assert _validate_typed_dict(data, TestMessage) is True

    def test_validate_typed_dict_missing_field(self):
        """Test validation with missing required field."""
        data = {"id": "test123"}  # Missing 'value' and 'optional_field'
        assert _validate_typed_dict(data, TestMessage) is False

    def test_validate_typed_dict_extra_field(self):
        """Test validation with extra field."""
        data = {
            "id": "test123",
            "value": 42,
            "optional_field": "hello",
            "extra_field": "not allowed"
        }
        assert _validate_typed_dict(data, TestMessage) is False

    def test_validate_typed_dict_wrong_type(self):
        """Test validation with wrong field type."""
        data = {
            "id": "test123",
            "value": "not_an_int",  # Should be int
            "optional_field": "hello"
        }
        assert _validate_typed_dict(data, TestMessage) is False

    def test_validate_typed_dict_not_dict(self):
        """Test validation with non-dict data."""
        assert _validate_typed_dict("not a dict", TestMessage) is False
        assert _validate_typed_dict(42, TestMessage) is False
        assert _validate_typed_dict(None, TestMessage) is False
