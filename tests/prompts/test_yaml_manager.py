"""Tests for YAMLPromptManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from graflow.prompts.exceptions import (
    PromptNotFoundError,
    PromptTypeError,
    PromptVersionNotFoundError,
)
from graflow.prompts.models import ChatPrompt, TextPrompt
from graflow.prompts.yaml_manager import YAMLPromptManager


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory with test files."""
    # Create main prompts.yaml
    prompts_yaml = tmp_path / "prompts.yaml"
    prompts_yaml.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}!"
      version: 2
      created_at: "2024-01-15"
    staging:
      content: "Hello {{name}} (staging)!"
      version: 1
      created_at: "2024-01-10"

farewell:
  type: text
  labels:
    production:
      content: "Goodbye {{name}}!"
      version: 1
""")

    # Create a chat prompt file
    chat_yaml = tmp_path / "chat.yaml"
    chat_yaml.write_text("""
assistant:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are {{assistant_type}}."
        - role: user
          content: "Hello!"
      version: 1
""")

    # Create subdirectory with prompts (virtual folders)
    customer_dir = tmp_path / "customer"
    customer_dir.mkdir()
    customer_yaml = customer_dir / "templates.yaml"
    customer_yaml.write_text("""
welcome:
  type: text
  labels:
    production:
      content: "Welcome to {{company}}, {{name}}!"
      version: 1
    latest:
      content: "Welcome aboard {{name}}!"
      version: 2
""")

    return tmp_path


@pytest.fixture
def manager(prompts_dir: Path) -> YAMLPromptManager:
    """Create a YAMLPromptManager with test prompts."""
    return YAMLPromptManager(prompts_dir=str(prompts_dir))


class TestYAMLPromptManagerInit:
    """Tests for YAMLPromptManager initialization."""

    def test_init_with_valid_directory(self, prompts_dir: Path):
        """Test initialization with valid prompts directory."""
        manager = YAMLPromptManager(prompts_dir=str(prompts_dir))
        assert manager.prompts_dir == prompts_dir

    def test_init_with_nonexistent_directory_works_as_noop(self, tmp_path: Path):
        """Test initialization with nonexistent directory works as no-op."""
        nonexistent = tmp_path / "nonexistent"

        # Should not raise - works as no-op
        manager = YAMLPromptManager(prompts_dir=str(nonexistent))

        # But get_text_prompt should raise PromptNotFoundError
        with pytest.raises(PromptNotFoundError):
            manager.get_text_prompt("any_prompt")

    def test_init_with_custom_cache_settings(self, prompts_dir: Path):
        """Test initialization with custom cache settings."""
        manager = YAMLPromptManager(
            prompts_dir=str(prompts_dir),
            cache_ttl=120,
            cache_maxsize=500,
        )
        assert manager._cache_ttl == 120

    def test_init_loads_all_prompts(self, prompts_dir: Path):
        """Test that initialization loads all prompts from directory."""
        manager = YAMLPromptManager(prompts_dir=str(prompts_dir))

        # Verify prompts are loaded by fetching them
        assert manager.get_text_prompt("greeting") is not None
        assert manager.get_text_prompt("farewell") is not None
        assert manager.get_chat_prompt("assistant") is not None
        assert manager.get_text_prompt("customer/welcome") is not None


class TestYAMLPromptManagerGetTextPrompt:
    """Tests for YAMLPromptManager.get_text_prompt()."""

    def test_get_text_prompt_by_label(self, manager: YAMLPromptManager):
        """Test getting text prompt by label."""
        prompt = manager.get_text_prompt("greeting", label="production")

        assert isinstance(prompt, TextPrompt)
        assert prompt.name == "greeting"
        assert prompt.label == "production"
        assert prompt.content == "Hello {{name}}!"
        assert prompt.version == 2

    def test_get_text_prompt_default_label(self, manager: YAMLPromptManager):
        """Test getting text prompt with default label (production)."""
        prompt = manager.get_text_prompt("greeting")

        assert prompt.label == "production"

    def test_get_text_prompt_staging_label(self, manager: YAMLPromptManager):
        """Test getting text prompt with staging label."""
        prompt = manager.get_text_prompt("greeting", label="staging")

        assert prompt.label == "staging"
        assert "(staging)" in prompt.content

    def test_get_text_prompt_by_version(self, manager: YAMLPromptManager):
        """Test getting text prompt by version number."""
        prompt = manager.get_text_prompt("greeting", version=1)

        assert prompt.version == 1
        assert "(staging)" in prompt.content

    def test_get_text_prompt_not_found(self, manager: YAMLPromptManager):
        """Test getting nonexistent prompt raises error."""
        with pytest.raises(PromptNotFoundError) as exc_info:
            manager.get_text_prompt("nonexistent")

        assert "nonexistent" in str(exc_info.value)

    def test_get_text_prompt_label_not_found(self, manager: YAMLPromptManager):
        """Test getting prompt with nonexistent label raises error."""
        with pytest.raises(PromptVersionNotFoundError) as exc_info:
            manager.get_text_prompt("greeting", label="nonexistent")

        assert "nonexistent" in str(exc_info.value)
        assert "production" in str(exc_info.value) or "staging" in str(exc_info.value)

    def test_get_text_prompt_version_not_found(self, manager: YAMLPromptManager):
        """Test getting prompt with nonexistent version raises error."""
        with pytest.raises(PromptVersionNotFoundError) as exc_info:
            manager.get_text_prompt("greeting", version=999)

        assert "999" in str(exc_info.value)

    def test_get_text_prompt_both_version_and_label_raises(self, manager: YAMLPromptManager):
        """Test that specifying both version and label raises error."""
        with pytest.raises(ValueError) as exc_info:
            manager.get_text_prompt("greeting", version=1, label="production")

        assert "Cannot specify both" in str(exc_info.value)

    def test_get_text_prompt_virtual_folder(self, manager: YAMLPromptManager):
        """Test getting text prompt from virtual folder."""
        prompt = manager.get_text_prompt("customer/welcome", label="production")

        assert prompt.name == "customer/welcome"
        assert "{{company}}" in prompt.content

    def test_get_text_prompt_type_mismatch_raises(self, manager: YAMLPromptManager):
        """Test getting chat prompt as text raises PromptTypeError."""
        with pytest.raises(PromptTypeError) as exc_info:
            manager.get_text_prompt("assistant", label="production")

        assert "chat prompt" in str(exc_info.value)
        assert "get_chat_prompt" in str(exc_info.value)


class TestYAMLPromptManagerGetChatPrompt:
    """Tests for YAMLPromptManager.get_chat_prompt()."""

    def test_get_chat_prompt(self, manager: YAMLPromptManager):
        """Test getting chat prompt."""
        prompt = manager.get_chat_prompt("assistant", label="production")

        assert isinstance(prompt, ChatPrompt)
        assert prompt.name == "assistant"
        assert isinstance(prompt.content, list)
        assert len(prompt.content) == 2
        assert prompt.content[0]["role"] == "system"

    def test_get_chat_prompt_type_mismatch_raises(self, manager: YAMLPromptManager):
        """Test getting text prompt as chat raises PromptTypeError."""
        with pytest.raises(PromptTypeError) as exc_info:
            manager.get_chat_prompt("greeting", label="production")

        assert "text prompt" in str(exc_info.value)
        assert "get_text_prompt" in str(exc_info.value)


class TestYAMLPromptManagerCaching:
    """Tests for YAMLPromptManager caching behavior with TLRUCache."""

    def test_cache_hit(self, manager: YAMLPromptManager):
        """Test that cached prompts are returned on subsequent calls."""
        # First call - loads from disk and caches
        prompt1 = manager.get_text_prompt("greeting", label="production")

        # Second call - should return from TLRUCache
        prompt2 = manager.get_text_prompt("greeting", label="production")

        # Both should return same content
        assert prompt1.content == prompt2.content
        # Verify cache has the entry
        assert manager._prompt_cache.contains(("greeting", "production", None))

    def test_cache_ttl_override(self, prompts_dir: Path):
        """Test that cache_ttl_seconds overrides default TTL."""
        manager = YAMLPromptManager(prompts_dir=str(prompts_dir), cache_ttl=300)

        # Load with custom TTL
        prompt = manager.get_text_prompt("greeting", label="production", cache_ttl_seconds=60)
        assert prompt.content == "Hello {{name}}!"

        # Verify cached
        assert manager._prompt_cache.contains(("greeting", "production", None))

    def test_file_modification_triggers_reload(self, tmp_path: Path):
        """Test that modified YAML files are reloaded after cache TTL expires."""
        import time

        # Create initial prompt file
        prompts_yaml = tmp_path / "prompts.yaml"
        prompts_yaml.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello v1"
      version: 1
""")

        # Use short TTL so cache expires quickly
        manager = YAMLPromptManager(prompts_dir=str(tmp_path), cache_ttl=1)

        # Load prompt (caches with 0.1s TTL)
        prompt1 = manager.get_text_prompt("greeting")
        assert prompt1.content == "Hello v1"

        # Wait for cache TTL to expire and ensure mtime changes
        time.sleep(2)

        # Modify the file
        prompts_yaml.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello v2"
      version: 2
""")

        # Next call should miss cache (TTL expired), detect modification, and reload
        prompt2 = manager.get_text_prompt("greeting")
        assert prompt2.content == "Hello v2"

    def test_file_mtime_tracked(self, manager: YAMLPromptManager):
        """Test that file modification times are tracked."""
        # Load a prompt to trigger file loading
        manager.get_text_prompt("greeting")

        # Verify file mtime is tracked
        assert len(manager._file_mtime) > 0
        for file_path, mtime in manager._file_mtime.items():
            assert file_path.exists()
            assert mtime > 0


class TestYAMLPromptManagerCollisionDetection:
    """Tests for collision detection during initialization."""

    def test_collision_detection_same_file(self, tmp_path: Path):
        """Test that YAML parser handles duplicate keys within same file."""
        # YAML parser will use last value for duplicate keys
        prompts_yaml = tmp_path / "prompts.yaml"
        prompts_yaml.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "First definition"
      version: 1

greeting:
  type: text
  labels:
    production:
      content: "Second definition"
      version: 2
""")

        # YAML parser uses last definition (standard YAML behavior)
        manager = YAMLPromptManager(prompts_dir=str(tmp_path))
        prompt = manager.get_text_prompt("greeting")
        assert prompt.content == "Second definition"

    def test_collision_detection_different_files(self, tmp_path: Path):
        """Test collision detection across different files logs warning and overwrites."""
        # First file
        file1 = tmp_path / "prompts1.yaml"
        file1.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "From file1"
      version: 1
""")

        # Second file with same prompt name and label
        file2 = tmp_path / "prompts2.yaml"
        file2.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "From file2"
      version: 2
""")

        # Construction succeeds (lazy loading)
        manager = YAMLPromptManager(prompts_dir=str(tmp_path))

        # Should not raise - just warns and overwrites
        prompt = manager.get_text_prompt("greeting")

        # One of the definitions wins (file loading order determines which)
        assert prompt.content in ["From file1", "From file2"]

    def test_no_collision_different_labels(self, tmp_path: Path):
        """Test that same prompt name with different labels is OK."""
        # First file with production label
        file1 = tmp_path / "prompts1.yaml"
        file1.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Production version"
      version: 1
""")

        # Second file with staging label (same prompt name, different label)
        file2 = tmp_path / "prompts2.yaml"
        file2.write_text("""
greeting:
  type: text
  labels:
    staging:
      content: "Staging version"
      version: 1
""")

        # Should not raise - different labels are OK
        manager = YAMLPromptManager(prompts_dir=str(tmp_path))
        # Verify both labels are accessible
        assert manager.get_text_prompt("greeting", label="production") is not None
        assert manager.get_text_prompt("greeting", label="staging") is not None


class TestYAMLPromptManagerIntegration:
    """Integration tests for YAMLPromptManager."""

    def test_render_loaded_text_prompt(self, manager: YAMLPromptManager):
        """Test rendering a loaded text prompt."""
        prompt = manager.get_text_prompt("greeting", label="production")
        result = prompt.render(name="Alice")

        assert result == "Hello Alice!"

    def test_render_chat_prompt(self, manager: YAMLPromptManager):
        """Test rendering a loaded chat prompt."""
        prompt = manager.get_chat_prompt("assistant", label="production")
        result = prompt.render(assistant_type="helpful bot")

        assert isinstance(result, list)
        assert result[0]["content"] == "You are helpful bot."

    def test_render_virtual_folder_prompt(self, manager: YAMLPromptManager):
        """Test rendering a prompt from virtual folder."""
        prompt = manager.get_text_prompt("customer/welcome", label="production")
        result = prompt.render(company="Acme", name="Bob")

        assert "Acme" in result
        assert "Bob" in result
