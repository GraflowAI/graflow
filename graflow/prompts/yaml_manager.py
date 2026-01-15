"""YAML-based prompt manager for local filesystem storage."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from yaml import safe_load

from graflow.prompts.base import PromptManager
from graflow.prompts.exceptions import PromptNotFoundError, PromptVersionNotFoundError
from graflow.prompts.models import ChatPrompt, PromptVersion, TextPrompt
from graflow.utils.cache import TLRUCache


class YAMLPromptManager(PromptManager):
    """YAML-based prompt manager with local filesystem storage.

    Caching Strategy:
    - TLRUCache stores prompts with per-entry TTL
    - File modification times are tracked; files are reloaded when modified
    - Default TTL from constructor, overridable per get_prompt() call
    """

    DEFAULT_TTL = 300  # 5 minutes
    DEFAULT_CACHE_MAXSIZE = 10000

    def __init__(
        self,
        prompts_dir: Optional[str] = None,
        cache_ttl: int = DEFAULT_TTL,
        cache_maxsize: int = DEFAULT_CACHE_MAXSIZE,
    ):
        """Initialize YAML prompt manager.

        Args:
            prompts_dir: Directory containing YAML prompt files.
                        If None, uses GRAFLOW_PROMPTS_DIR env var.
                        If env var not set, defaults to "./prompts".
            cache_ttl: Default cache TTL in seconds (default: 300).
                      Set to 0 for no expiration. Can be overridden per get_prompt() call.
            cache_maxsize: Maximum number of cached prompt entries (default: 10000).

        Note:
            Prompts are loaded lazily on first get_prompt() call.
            YAML files are reloaded automatically when modified.
            Full prompt name = subdir/yaml_key (relative to prompts_dir root).
        """
        if prompts_dir is None:
            prompts_dir = os.getenv("GRAFLOW_PROMPTS_DIR", "./prompts")

        self.prompts_dir: Path = Path(prompts_dir).resolve()

        # Default cache TTL in seconds (0 = never expires)
        self._cache_ttl = cache_ttl

        # TLRUCache for prompt storage
        # Key: (name, label, version), Value: PromptVersion
        self._prompt_cache: TLRUCache[Tuple[str, str, int], PromptVersion] = TLRUCache(maxsize=cache_maxsize)

        # Track file modification times: {file_path: mtime}
        self._file_mtime: Dict[Path, float] = {}

        # Track loaded prompts: {prompt_name: {label: file_path}}
        # Used for: existence check, available labels, file reload tracking
        self._loaded_prompt_files: Dict[str, Dict[str, str]] = {}

        # Track latest version per (name, label): {(name, label): latest_version}
        self._latest_version: Dict[Tuple[str, str], int] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> PromptVersion:
        """Get prompt from YAML files.

        Semantics:
            - Label is mandatory in YAML (e.g., "production", "staging").
            - Version is optional in YAML; defaults to 1 if not specified.
            - Each (name, label) pair has its own version tracking.
            - When version arg is omitted, returns the latest version for the label.

        Args:
            name: Prompt name (format: "prompt_key" or "subdir/prompt_key").
            version: Numeric version (optional, uses latest version for label if not specified).
            label: Version label (default: "production").
            cache_ttl_seconds: Cache TTL override for this entry.
                              If None, uses constructor's cache_ttl.
                              Set to 0 for no expiration.

        Returns:
            PromptVersion instance (TextPrompt or ChatPrompt).

        Raises:
            PromptNotFoundError: If prompt not found in any directory.
            PromptVersionNotFoundError: If version/label not found.
        """
        # Determine target label (default: "production")
        target_label = label if label is not None else "production"

        # Determine TTL (cache_ttl_seconds overrides default _cache_ttl)
        ttl = cache_ttl_seconds if cache_ttl_seconds is not None else self._cache_ttl

        # Load prompt file if required (updates cache)
        self._load_prompt_file(name, ttl)

        # Determine target version (use latest if not specified)
        target_version: int
        if version is not None:
            target_version = version
        else:
            latest = self._latest_version.get((name, target_label))
            if latest is not None:
                target_version = latest
            else:
                # Prompt or label not found - will raise error below
                target_version = 1  # Default for error path

        # Build cache key and check cache
        cache_key = (name, target_label, target_version)
        refresh_ttl = ttl if cache_ttl_seconds is not None else None
        cached = self._prompt_cache.get(cache_key, ttl_seconds=refresh_ttl)
        if cached is not None:
            return cached

        # If cache entries expired, reload prompt files and try again
        if name in self._loaded_prompt_files:
            for file_path_str in set(self._loaded_prompt_files[name].values()):
                self._reload_file(Path(file_path_str), ttl)
            # Refresh target_version after reload
            if version is None:
                latest = self._latest_version.get((name, target_label))
                if latest is not None:
                    target_version = latest
                    cache_key = (name, target_label, target_version)
            cached = self._prompt_cache.get(cache_key)
            if cached is not None:
                return cached

        # Determine error type: prompt not found vs label/version not found
        if name not in self._loaded_prompt_files:
            raise PromptNotFoundError(f"Prompt not found: '{name}'")

        # Prompt exists but label/version not found
        available_labels = list(self._loaded_prompt_files[name].keys())
        if version is not None:
            raise PromptVersionNotFoundError(
                f"Version {version} not found for prompt '{name}' with label '{target_label}'. "
                f"Available labels: {available_labels}"
            )
        raise PromptVersionNotFoundError(
            f"Label '{target_label}' not found for prompt '{name}'. Available labels: {available_labels}"
        )

    # -------------------------------------------------------------------------
    # Prompt loading
    # -------------------------------------------------------------------------

    def _load_prompt_file(self, name: str, ttl: int) -> None:
        """Load prompt file if needed, reload if modified.

        Args:
            name: Prompt name (format: "prompt_key" or "subdir/prompt_key").
            ttl: Cache TTL in seconds for new entries.
        """
        if not self.prompts_dir.exists():
            return

        # Extract subdirectory from prompt name (e.g., "customer/welcome" -> "customer")
        # rsplit("/", 1)[0] splits from right once, getting the directory path portion
        subdir = name.rsplit("/", 1)[0] if "/" in name else ""
        target_dir = self.prompts_dir / subdir if subdir else self.prompts_dir

        if not target_dir.exists():
            return

        # Check if prompt's source file was modified or removed and reload
        if name in self._loaded_prompt_files:
            for file_path_str in set(self._loaded_prompt_files[name].values()):
                file_path = Path(file_path_str)
                cached_mtime = self._file_mtime.get(file_path)
                if cached_mtime is None:
                    continue
                if not file_path.exists():
                    self._reload_file(file_path, ttl)
                    continue
                if file_path.stat().st_mtime > cached_mtime:
                    self._reload_file(file_path, ttl)

        # Load new files and reload modified files in the target directory
        yaml_files = list(target_dir.glob("*.yaml")) + list(target_dir.glob("*.yml"))
        for yaml_file in yaml_files:
            cached_mtime = self._file_mtime.get(yaml_file)
            if cached_mtime is None:
                self._parse_yaml_file(yaml_file, ttl)
            elif yaml_file.stat().st_mtime > cached_mtime:
                self._reload_file(yaml_file, ttl)

    # -------------------------------------------------------------------------
    # YAML parsing
    # -------------------------------------------------------------------------

    def _parse_yaml_file(self, file_path: Path, ttl: int) -> None:
        """Parse a single YAML file and register prompts.

        Args:
            file_path: Path to YAML file.
            ttl: Cache TTL in seconds.
        """
        self._file_mtime[file_path] = file_path.stat().st_mtime

        with open(file_path, encoding="utf-8") as f:
            data = safe_load(f)

        if not isinstance(data, dict):
            return

        # Calculate virtual folder path relative to base directory
        relative_path = file_path.parent.relative_to(self.prompts_dir)
        folder_prefix = "" if str(relative_path) == "." else str(relative_path).replace("\\", "/") + "/"

        # Process each prompt in the file
        for prompt_key, prompt_data in data.items():
            if not isinstance(prompt_data, dict):
                continue

            full_name = folder_prefix + prompt_key
            prompt_type = prompt_data.get("type", "text")
            labels = prompt_data.get("labels", {})
            if not isinstance(labels, dict):
                continue

            for label_name, label_data in labels.items():
                if not isinstance(label_data, dict):
                    continue
                prompt_version = self._create_prompt_version(prompt_type, full_name, label_name, label_data)

                # Version is always set (defaults to 1 in _create_prompt_version)
                version_num = prompt_version.version
                assert version_num is not None  # For type checker

                # Store in cache with explicit (name, label, version) key
                self._prompt_cache.set((full_name, label_name, version_num), prompt_version, ttl_seconds=ttl)

                # Update latest version tracking (always overwrite since YAML is source of truth)
                self._latest_version[(full_name, label_name)] = version_num

                # Track file-to-prompt mapping for reload
                if full_name not in self._loaded_prompt_files:
                    self._loaded_prompt_files[full_name] = {}
                self._loaded_prompt_files[full_name][label_name] = str(file_path)

    def _create_prompt_version(
        self, prompt_type: str, full_name: str, label_name: str, label_data: Dict[str, Any]
    ) -> PromptVersion:
        """Create a PromptVersion instance from label data."""
        # Version defaults to 1 if not specified in YAML
        version_num: int = label_data.get("version", 1)
        created_at: Optional[str] = label_data.get("created_at")
        metadata: Dict[str, Any] = label_data.get("metadata", {})

        if prompt_type == "chat":
            content_chat: List[Dict[str, Any]] = label_data.get("content", [])
            return ChatPrompt(
                name=full_name,
                content=content_chat,
                version=version_num,
                label=label_name,
                created_at=created_at,
                metadata=metadata,
            )
        content_text: str = label_data.get("content", "")
        return TextPrompt(
            name=full_name,
            content=content_text,
            version=version_num,
            label=label_name,
            created_at=created_at,
            metadata=metadata,
        )

    # -------------------------------------------------------------------------
    # File reload
    # -------------------------------------------------------------------------

    def _reload_file(self, file_path: Path, ttl: int) -> None:
        """Reload a single file. Cache entries are overwritten by _parse_yaml_file()."""
        if file_path.exists():
            self._parse_yaml_file(file_path, ttl)
        else:
            self._file_mtime.pop(file_path, None)
