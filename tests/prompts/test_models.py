"""Tests for TextPrompt and ChatPrompt models."""

from __future__ import annotations

import pytest

from graflow.prompts.exceptions import PromptCompilationError
from graflow.prompts.models import ChatPrompt, PromptVersion, TextPrompt


class TestPromptVersionBase:
    """Tests for PromptVersion base class."""

    def test_base_class_initialization(self):
        """Test base class initialization."""
        pv = PromptVersion(
            name="test",
            version=1,
            label="production",
        )

        assert pv.name == "test"
        assert pv.version == 1
        assert pv.label == "production"
        assert pv.created_at is None
        assert pv.metadata == {}

    def test_optional_fields(self):
        """Test that optional fields have correct defaults."""
        pv = PromptVersion(name="minimal")

        assert pv.version is None
        assert pv.label is None
        assert pv.created_at is None
        assert pv.metadata == {}

    def test_metadata_field(self):
        """Test metadata field storage."""
        metadata = {"author": "test", "tags": ["greeting", "customer"]}
        pv = PromptVersion(
            name="with_metadata",
            metadata=metadata,
        )

        assert pv.metadata == metadata


class TestTextPromptInit:
    """Tests for TextPrompt initialization."""

    def test_text_prompt_initialization(self):
        """Test creating a text prompt."""
        tp = TextPrompt(
            name="greeting",
            content="Hello {{name}}!",
            version=1,
            label="production",
        )

        assert tp.name == "greeting"
        assert tp.content == "Hello {{name}}!"
        assert tp.version == 1
        assert tp.label == "production"

    def test_text_prompt_is_prompt_version(self):
        """Test that TextPrompt is a PromptVersion."""
        tp = TextPrompt(name="test", content="Hello")
        assert isinstance(tp, PromptVersion)


class TestTextPromptRender:
    """Tests for TextPrompt.render() method."""

    def test_render_single_variable(self):
        """Test rendering text prompt with single variable."""
        tp = TextPrompt(
            name="greeting",
            content="Hello {{name}}!",
        )

        result = tp.render(name="Alice")
        assert result == "Hello Alice!"

    def test_render_multiple_variables(self):
        """Test rendering text prompt with multiple variables."""
        tp = TextPrompt(
            name="greeting",
            content="Hello {{name}}, welcome to {{company}}!",
        )

        result = tp.render(name="Bob", company="Acme Corp")
        assert result == "Hello Bob, welcome to Acme Corp!"

    def test_render_no_variables(self):
        """Test rendering text prompt without variables."""
        tp = TextPrompt(
            name="static",
            content="This is a static message.",
        )

        result = tp.render()
        assert result == "This is a static message."

    def test_render_missing_variable_raises_error(self):
        """Test that missing variables raise PromptCompilationError."""
        tp = TextPrompt(
            name="greeting",
            content="Hello {{name}}!",
        )

        with pytest.raises(PromptCompilationError) as exc_info:
            tp.render()  # Missing 'name' variable

        assert "greeting" in str(exc_info.value)

    def test_render_extra_variables_ignored(self):
        """Test that extra variables are silently ignored."""
        tp = TextPrompt(
            name="greeting",
            content="Hello {{name}}!",
        )

        result = tp.render(name="Eve", extra="ignored")
        assert result == "Hello Eve!"

    def test_render_with_special_characters(self):
        """Test rendering with special characters in variables."""
        tp = TextPrompt(
            name="special",
            content="Message: {{text}}",
        )

        result = tp.render(text="Hello <world> & 'friends'!")
        assert result == "Message: Hello <world> & 'friends'!"

    def test_render_with_unicode(self):
        """Test rendering with unicode characters."""
        tp = TextPrompt(
            name="unicode",
            content="Greeting: {{greeting}}",
        )

        result = tp.render(greeting="こんにちは 🎉")
        assert result == "Greeting: こんにちは 🎉"

    def test_render_with_multiline_content(self):
        """Test rendering multiline content."""
        tp = TextPrompt(
            name="multiline",
            content="Dear {{name}},\n\nWelcome to {{place}}!\n\nBest regards",
        )

        result = tp.render(name="Frank", place="the team")
        assert "Dear Frank," in result
        assert "Welcome to the team!" in result


class TestChatPromptInit:
    """Tests for ChatPrompt initialization."""

    def test_chat_prompt_initialization(self):
        """Test creating a chat prompt."""
        messages = [
            {"role": "system", "content": "You are {{assistant_type}}."},
            {"role": "user", "content": "Hello!"},
        ]
        cp = ChatPrompt(
            name="chat_greeting",
            content=messages,
            version=2,
            label="staging",
        )

        assert cp.name == "chat_greeting"
        assert cp.content == messages
        assert cp.version == 2
        assert cp.label == "staging"

    def test_chat_prompt_is_prompt_version(self):
        """Test that ChatPrompt is a PromptVersion."""
        cp = ChatPrompt(name="test", content=[{"role": "user", "content": "Hi"}])
        assert isinstance(cp, PromptVersion)


class TestChatPromptRender:
    """Tests for ChatPrompt.render() method."""

    def test_render_chat_prompt(self):
        """Test rendering chat prompt with variables."""
        messages = [
            {"role": "system", "content": "You are {{assistant_type}}."},
            {"role": "user", "content": "My name is {{user_name}}."},
        ]
        cp = ChatPrompt(
            name="chat",
            content=messages,
        )

        result = cp.render(assistant_type="a helpful assistant", user_name="Charlie")

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are a helpful assistant."
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "My name is Charlie."

    def test_render_chat_prompt_preserves_all_fields(self):
        """Test that rendering preserves all message fields."""
        messages = [
            {
                "role": "assistant",
                "content": "Hello {{name}}!",
                "name": "bot",
                "function_call": None,
            },
        ]
        cp = ChatPrompt(
            name="chat",
            content=messages,
        )

        result = cp.render(name="Dave")

        assert isinstance(result, list)
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hello Dave!"
        assert result[0]["name"] == "bot"
        assert result[0]["function_call"] is None

    def test_render_missing_variable_raises_error(self):
        """Test that missing variables raise PromptCompilationError."""
        cp = ChatPrompt(
            name="chat",
            content=[{"role": "user", "content": "Hello {{name}}!"}],
        )

        with pytest.raises(PromptCompilationError) as exc_info:
            cp.render()  # Missing 'name' variable

        assert "chat" in str(exc_info.value)


class TestPromptRepr:
    """Tests for prompt string representation."""

    def test_text_prompt_repr(self):
        """Test string representation of text prompt."""
        tp = TextPrompt(
            name="test",
            content="Hello",
            version=1,
            label="prod",
        )

        str_repr = str(tp)
        assert "test" in str_repr
        assert "TextPrompt" in str_repr

    def test_chat_prompt_repr(self):
        """Test string representation of chat prompt."""
        cp = ChatPrompt(
            name="chat_test",
            content=[{"role": "user", "content": "Hi"}],
        )

        str_repr = str(cp)
        assert "chat_test" in str_repr
        assert "ChatPrompt" in str_repr
