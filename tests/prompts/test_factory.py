"""Tests for PromptManagerFactory."""

from __future__ import annotations

from pathlib import Path

import pytest

from graflow.prompts.factory import PromptManagerFactory
from graflow.prompts.yaml_manager import YAMLPromptManager


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    """Create a temporary prompts directory with test files."""
    prompts_yaml = tmp_path / "prompts.yaml"
    prompts_yaml.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}!"
      version: 1
""")
    return tmp_path


class TestPromptManagerFactory:
    """Tests for PromptManagerFactory."""

    def test_create_yaml_manager(self, prompts_dir: Path) -> None:
        """Test creating YAML manager via factory."""
        manager = PromptManagerFactory.create(
            backend="yaml",
            prompts_dir=str(prompts_dir),
        )

        assert isinstance(manager, YAMLPromptManager)

    def test_create_default_backend(self, prompts_dir: Path) -> None:
        """Test that default backend is yaml."""
        manager = PromptManagerFactory.create(prompts_dir=str(prompts_dir))

        assert isinstance(manager, YAMLPromptManager)

    def test_create_unknown_backend_raises(self) -> None:
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            PromptManagerFactory.create(backend="unknown")

        assert "unknown" in str(exc_info.value).lower()

    def test_factory_passes_kwargs_to_manager(self, prompts_dir: Path) -> None:
        """Test that factory passes kwargs to manager constructor."""
        # Should not raise - kwargs passed to manager
        manager = PromptManagerFactory.create(
            backend="yaml",
            prompts_dir=str(prompts_dir),
            cache_ttl=120,
        )

        assert isinstance(manager, YAMLPromptManager)


class TestPromptManagerFactoryLangfuse:
    """Tests for Langfuse backend in factory (conditional)."""

    def test_langfuse_backend_conditional(self) -> None:
        """Test that langfuse backend availability depends on installation."""
        try:
            from graflow.prompts.langfuse_manager import LANGFUSE_AVAILABLE

            if LANGFUSE_AVAILABLE:
                # Should not raise if langfuse is installed
                # (but will fail if credentials are not set)
                pass
        except ImportError:
            # Langfuse not installed - that's OK
            pass
