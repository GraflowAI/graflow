"""Tests for LLM client."""

import pytest

from graflow.llm.client import LLMClient, make_message

# Skip tests if litellm is not available
pytest.importorskip("litellm")


class TestLLMClient:
    """Test LLMClient class."""

    def test_client_creation(self):
        """Test basic client creation."""
        client = LLMClient(model="gpt-4o-mini")

        assert client.model == "gpt-4o-mini"
        assert client.default_params == {}

    def test_client_with_default_params(self):
        """Test client with default parameters."""
        client = LLMClient(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024
        )

        assert client.model == "gpt-4o-mini"
        assert client.default_params["temperature"] == 0.7
        assert client.default_params["max_tokens"] == 1024


class TestMakeMessage:
    """Test make_message helper function."""

    def test_make_message_user(self):
        """Test creating user message."""
        msg = make_message("user", "Hello!")

        assert msg == {"role": "user", "content": "Hello!"}

    def test_make_message_system(self):
        """Test creating system message."""
        msg = make_message("system", "You are helpful.")

        assert msg == {"role": "system", "content": "You are helpful."}

    def test_make_message_assistant(self):
        """Test creating assistant message."""
        msg = make_message("assistant", "How can I help?")

        assert msg == {"role": "assistant", "content": "How can I help?"}
