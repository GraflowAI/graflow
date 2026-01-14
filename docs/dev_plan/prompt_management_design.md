# Prompt Management Module Design

## Overview

Design document for implementing prompt management in Graflow with two backend implementations:
1. **YAMLPromptManager** (default): Local filesystem storage with YAML files
2. **LangfusePromptManager**: Langfuse cloud/server integration for versioned prompt management

### Purpose

- Centralized prompt management for LLM workflows
- Version control for prompts (production, staging, experiments)
- Template variable substitution with Jinja2
- Seamless integration between local development (YAML) and production (Langfuse)
- Support for both single prompts and conversation prompts

### Key Design Decisions

1. **Template Format**: `{{variable}}` (Mustache/Jinja2 syntax)
2. **Dependency**: Jinja2 (already in core dependencies)
3. **Virtual Folders**: Slash-based prompt names (e.g., `customer/greeting/welcome`)
4. **Directory Mapping**: Filesystem directories map to Langfuse-compatible virtual folders
5. **Versioning**: Labels (`production`, `staging`, `latest`) with numeric version tracking
6. **API Pattern**: `get_prompt()` returns `PromptVersion` object, `.render()` compiles template
7. **Caching**: Per-call `ttl_seconds` parameter with smart defaults
8. **Configuration**:
   - YAML: `prompts_dir` optional with `GRAFLOW_PROMPTS_DIR` env var fallback
   - Langfuse: `fetch_timeout_seconds` and `max_retries` configured at manager initialization
9. **Langfuse Compatibility**: Compatible with Langfuse SDK (manager-level retry and timeout configuration)
10. **Backend Abstraction**: Unified `PromptVersion` return type from both YAML and Langfuse backends

---

## Architecture

### Module Structure

```
graflow/prompts/
├── __init__.py              # Public API exports
├── base.py                  # PromptManager abstract base class
├── models.py                # Data models (Prompt, PromptVersion)
├── yaml_manager.py          # YAML backend implementation (default)
├── langfuse_manager.py      # Langfuse backend implementation
├── factory.py               # PromptManagerFactory
└── exceptions.py            # Prompt-specific exceptions
```

### Class Hierarchy

```
PromptManager (ABC)
├── YAMLPromptManager       # Local YAML files
└── LangfusePromptManager   # Langfuse cloud/server
```

### Integration with Graflow

```
ExecutionContext
  └── prompt_manager: PromptManager
      └── get_prompt(name, version, label, ttl_seconds) → PromptVersion
          └── render(**variables) → str | List[Dict]
```

---

## YAML Format Specification

### Basic Structure

```yaml
prompt-name:
  type: text  # or 'chat'
  labels:
    label-name:
      content: "Template with {{variables}}"
      created_at: "2024-01-15T10:00:00"
      version: 3  # Numeric version ID
```

### Text Prompt Example

```yaml
customer-greeting:
  type: text
  labels:
    production:
      content: "Hello {{customer_name}}, welcome to {{product}}!"
      created_at: "2024-01-15T10:00:00"
      version: 3

    staging:
      content: "Hi {{customer_name}}! Welcome to {{product}} beta."
      created_at: "2024-01-20T14:00:00"
      version: 4

    latest:
      content: "Hi {{customer_name}}! Welcome to {{product}} beta."
      created_at: "2024-01-20T14:00:00"
      version: 4
```

### Chat Prompt Example

```yaml
interview:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are an expert in {{domain}}."
        - role: user
          content: "Interview me about {{topic}}."
      created_at: "2024-01-15T14:30:00"
      version: 2

    staging:
      content:
        - role: system
          content: "You are a world-class expert in {{domain}} with 20 years of experience."
        - role: user
          content: "Conduct a detailed interview about {{topic}}."
      created_at: "2024-01-18T10:00:00"
      version: 3
```

### Key Elements

- **Prompt name**: Top-level YAML key (supports hyphens, underscores)
- **type**: `text` (string output) or `chat` (message list output)
- **labels**: Dictionary of label names to prompt versions
- **content**: Template string (text) or message list (chat)
- **created_at**: ISO timestamp
- **version**: Numeric identifier (1, 2, 3...)
- **Template syntax**: `{{variable}}` for Jinja2 substitution

---

## Directory Structure & Virtual Folders

### Concept

Langfuse uses **virtual folders** via slashes in prompt names (e.g., `customer/greeting/welcome`). Our YAML backend maps filesystem directories to these virtual folders for Langfuse compatibility.

### Mapping Rule

**Full prompt name = `directory_path` + `/` + `yaml_key`**

### File Discovery Rules

The YAML loader scans the prompts directory with the following rules:

1. **Recursive scan**: All subdirectories are searched
2. **File matching**: All files matching `*.yaml` or `*.yml` are loaded
3. **Filename irrelevant**: Filename does NOT affect prompt names (only directory path matters)
4. **Multiple prompts per file**: Each YAML file can contain multiple top-level keys
5. **Collision detection**: If two files define the same prompt name with the same label, raises `PromptConfigurationError`
6. **Collision scope**: Checked at both file level (within single file) and cross-file level (across multiple files)

**Collision Detection Examples:**

```python
# Case 1: Same name+label in ONE file - ERROR
# prompts/test.yaml
greeting:
  labels:
    production: {...}

greeting:  # ERROR: Duplicate key 'greeting'
  labels:
    production: {...}

# Case 2: Same name+label in DIFFERENT files - ERROR
# prompts/prompt1.yaml
greeting:
  labels:
    production: {...}

# prompts/prompt2.yaml
greeting:
  labels:
    production: {...}  # ERROR: PromptConfigurationError

# Case 3: Same name but DIFFERENT labels - OK
# prompts/prompt1.yaml
greeting:
  labels:
    production: {...}

# prompts/prompt2.yaml
greeting:
  labels:
    staging: {...}  # OK: Different label
```

### Directory Examples

#### Simple (Root Level)

```
prompts/
  prompts.yaml
```

```yaml
# prompts/prompts.yaml
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}!"
```

**Access**: `pm.get_prompt("greeting")`

---

#### One-Level Folders

```
prompts/
  customer/
    prompts.yaml
  internal/
    prompts.yaml
```

```yaml
# prompts/customer/prompts.yaml
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}, welcome to {{product}}!"

order-confirmation:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are a customer service agent."
        - role: user
          content: "Confirm order {{order_id}}."
```

**Access**:
- `pm.get_prompt("customer/greeting")`
- `pm.get_prompt("customer/order-confirmation")`

---

#### Multi-Level Folders

```
prompts/
  customer/
    onboarding/
      prompts.yaml
    support/
      prompts.yaml
  internal/
    alerts/
      critical/
        prompts.yaml
```

```yaml
# prompts/customer/onboarding/prompts.yaml
welcome:
  type: text
  labels:
    production:
      content: "Welcome {{name}}!"

tour-start:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are a tour guide for {{product}}."
```

**Access**:
- `pm.get_prompt("customer/onboarding/welcome")`
- `pm.get_prompt("customer/onboarding/tour-start")`
- `pm.get_prompt("internal/alerts/critical/system-down")`

---

#### Mixed (Root + Folders)

```
prompts/
  simple-greeting.yaml     # Root level
  customer/
    prompts.yaml           # customer/* prompts
  internal/
    admin/
      prompts.yaml         # internal/admin/* prompts
```

**Access**:
- `pm.get_prompt("simple-greeting")` (root)
- `pm.get_prompt("customer/greeting")` (one level)
- `pm.get_prompt("internal/admin/alert")` (two levels)

---

### Mapping Table

| File Path | YAML Key | Full Prompt Name |
|-----------|----------|------------------|
| `prompts/prompts.yaml` | `greeting` | `greeting` |
| `prompts/customer/prompts.yaml` | `greeting` | `customer/greeting` |
| `prompts/customer/onboarding/prompts.yaml` | `welcome` | `customer/onboarding/welcome` |
| `prompts/internal/admin/alerts.yaml` | `system-down` | `internal/admin/system-down` |

---

## API Design

### PromptManager Abstract Base Class

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict

class PromptManager(ABC):
    """Abstract base class for prompt management backends."""

    @abstractmethod
    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> PromptVersion:
        """Get prompt by name with optional version/label.

        Args:
            name: Prompt name (can include folder path, e.g., "customer/greeting")
            version: Numeric version (1, 2, 3...)
            label: Version label ("production", "staging", "latest")
            ttl_seconds: Cache TTL in seconds
                        None = use default (300 seconds)
                        -1 = no caching (always fetch fresh)
                        0 = infinite cache (never expires)
                        >0 = cache for N seconds

        Returns:
            PromptVersion object with render() method

        Note:
            - If neither version nor label specified, uses "production" label
            - If both version and label specified, raises ValueError (ambiguous request)
            - Labels are the primary access method; version numbers are for direct access or audit
            - Fetch timeout and max retries are configured at manager initialization (Langfuse only)
        """
        pass

    @abstractmethod
    def list_prompts(self) -> List[str]:
        """List all available prompt names."""
        pass

    @abstractmethod
    def list_labels(self, name: str) -> List[str]:
        """List all labels for a prompt."""
        pass
```

### PromptVersion Model

```python
from __future__ import annotations

from typing import Union, List, Dict, Any, Optional, Literal
from jinja2 import Template

class PromptVersion:
    """Represents a specific version of a prompt."""

    def __init__(
        self,
        name: str,
        content: Union[str, List[Dict[str, Any]]],
        type: Literal['text', 'chat'],
        version: Optional[int] = None,
        label: Optional[str] = None,
        created_at: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.content = content
        self.type = type
        self.version = version
        self.label = label
        self.created_at = created_at
        self.metadata = metadata or {}

    def render(self, **variables) -> Union[str, List[Dict[str, Any]]]:
        """Render template with variables using Jinja2 StrictUndefined mode.

        Args:
            **variables: Template variables to substitute

        Returns:
            Compiled prompt (string for 'text', message list for 'chat')

        Raises:
            PromptCompilationError: If template rendering fails (missing variables, syntax errors)

        Example:
            ```python
            # Text prompt
            prompt = pm.get_prompt("greeting")
            result = prompt.render(name="Alice", product="Graflow")
            # result: "Hello Alice, welcome to Graflow!"

            # Chat prompt
            prompt = pm.get_prompt("interview")
            messages = prompt.render(domain="AI", topic="transformers")
            # messages: [
            #   {'role': 'system', 'content': 'You are an expert in AI.'},
            #   {'role': 'user', 'content': 'Interview me about transformers.'}
            # ]
            ```
        """
        from jinja2 import Environment, StrictUndefined, TemplateError

        # Use StrictUndefined to raise errors on missing variables
        env = Environment(undefined=StrictUndefined)

        try:
            if self.type == 'text':
                template = env.from_string(self.content)
                return template.render(**variables)
            elif self.type == 'chat':
                # Compile each message content
                compiled_messages = []
                for msg in self.content:
                    # Only render 'content' field if it's a string
                    if isinstance(msg.get('content'), str):
                        template = env.from_string(msg['content'])
                        content = template.render(**variables)
                    else:
                        content = msg.get('content')

                    # Preserve all fields (role, content, tool_calls, etc.)
                    compiled_msg = {**msg, 'content': content}
                    compiled_messages.append(compiled_msg)
                return compiled_messages
            else:
                raise ValueError(f"Unknown prompt type: {self.type}")
        except TemplateError as e:
            raise PromptCompilationError(
                f"Failed to compile prompt '{self.name}': {e}"
            ) from e
```

---

## Versioning: Labels vs Versions

### Concepts

- **Label**: Primary access method - string identifier ("production", "staging", "latest", etc.)
- **Version**: Numeric metadata (1, 2, 3...) embedded in each label entry for tracking/audit
- **Each label contains**: full prompt content + version number + metadata
- **Labels are NOT pointers**: Each label entry is self-contained with its own content
- **Version numbers**: Track iteration history, allow direct access, enable audit trail

### Design Rationale

In YAML, each label is a complete prompt specification (content + metadata). This differs from a traditional "label points to version" model because:
1. YAML files are simple and self-contained
2. No separate version storage needed
3. Easy to copy/paste between labels
4. Version number is metadata for tracking which iteration this represents

### Access Rules

1. **By label only**: `get_prompt(name, label="production")` - Primary method
2. **By version only**: `get_prompt(name, version=3)` - Direct access for audit/rollback
3. **Neither**: `get_prompt(name)` - Defaults to `label="production"`
4. **Both**: `get_prompt(name, label="X", version=Y)` - **Raises ValueError** (ambiguous)

### Special Labels

- **`production`**: Default label when none specified
- **`latest`**: Conventionally points to newest version
- **Custom labels**: `staging`, `experiment-a`, `team-alpha`, etc.

### Version Evolution Example

```yaml
welcome-message:
  type: text
  labels:
    v1-archive:
      content: "Welcome {{name}}"
      version: 1
      created_at: "2024-01-01T10:00:00"

    v2-archive:
      content: "Hello {{name}}, welcome!"
      version: 2
      created_at: "2024-01-05T10:00:00"

    production:
      content: "Hello {{name}}, welcome to {{product}}!"
      version: 3
      created_at: "2024-01-15T10:00:00"

    experiment-a:
      content: "Hey {{name}} 👋 Welcome to {{product}}!"
      version: 4
      created_at: "2024-01-20T14:00:00"

    latest:
      content: "Hey {{name}} 👋 Welcome to {{product}}!"
      version: 4
      created_at: "2024-01-20T14:00:00"
```

### Access Patterns

```python
# By label (most common)
prompt = pm.get_prompt("welcome-message", label="production")  # version 3
prompt = pm.get_prompt("welcome-message", label="experiment-a")  # version 4

# By version number
prompt = pm.get_prompt("welcome-message", version=3)

# Default (uses "production" label)
prompt = pm.get_prompt("welcome-message")  # version 3

# Check version info
print(prompt.version)  # 3
print(prompt.label)    # "production"
```

---

## Caching Strategy

### Backend-Specific Caching

**YAML Backend**: Custom cache implementation with manual TTL management
**Langfuse Backend**: Delegates to Langfuse SDK's built-in caching via `cache_ttl_seconds` parameter

### Per-Call TTL Control

Each `get_prompt()` call can specify cache TTL independently:

```python
def get_prompt(
    self,
    name: str,
    *,
    version: Optional[int] = None,
    label: Optional[str] = None,
    ttl_seconds: Optional[int] = None,  # Per-call cache control
) -> PromptVersion:
```

### TTL Values

| Value | Behavior |
|-------|----------|
| `None` | Use default TTL (300 seconds / 5 minutes) |
| `-1` | No caching (bypass cache, always fetch fresh) |
| `0` | Infinite cache (never expires) |
| `> 0` | Cache for N seconds |

### Cache Implementation (YAML Backend Only)

**Note**: This cache implementation is only for YAML backend. Langfuse backend uses SDK's built-in caching.

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple
from cachetools import TTLCache, Cache

class YAMLPromptManager(PromptManager):
    DEFAULT_TTL = 300  # 5 minutes

    def __init__(
        self,
        prompts_dir: Optional[str] = None,
        cache_maxsize: int = 1000
    ):
        import os

        # Resolve prompts directory
        if prompts_dir is None:
            prompts_dir = os.getenv("GRAFLOW_PROMPTS_DIR", "./prompts")

        self.prompts_dir = Path(prompts_dir).resolve()
        self.cache_maxsize = cache_maxsize

        # Use cachetools for LRU eviction
        # We use Cache (no TTL) and manage expiry manually per-key
        self._cache: Dict[Tuple, Tuple[PromptVersion, Optional[float]]] = {}

        # Track loaded prompts for collision detection
        # Structure: {prompt_name: {label: file_path}}
        self._loaded_prompts: Dict[str, Dict[str, str]] = {}

        # Validate directory exists
        if not self.prompts_dir.exists():
            raise ValueError(f"Prompts directory not found: {self.prompts_dir}")

        # Load all prompts (raises PromptConfigurationError on collision)
        self._load_all_prompts()

    def _load_all_prompts(self) -> None:
        """Load all prompts from directory with collision detection.

        Raises:
            PromptConfigurationError: If duplicate prompt name+label found in multiple files
        """
        from yaml import safe_load

        # Recursively find all YAML files
        for yaml_file in self.prompts_dir.rglob("*.yaml"):
            self._load_prompt_file(yaml_file)
        for yaml_file in self.prompts_dir.rglob("*.yml"):
            self._load_prompt_file(yaml_file)

    def _load_prompt_file(self, file_path: Path) -> None:
        """Load prompts from a single YAML file with collision detection.

        Args:
            file_path: Path to YAML file

        Raises:
            PromptConfigurationError: If duplicate prompt name+label detected
        """
        from yaml import safe_load

        with open(file_path, "r", encoding="utf-8") as f:
            data = safe_load(f)

        if not isinstance(data, dict):
            return

        # Calculate virtual folder path
        relative_path = file_path.parent.relative_to(self.prompts_dir)
        if str(relative_path) == ".":
            folder_prefix = ""
        else:
            folder_prefix = str(relative_path).replace("\\", "/") + "/"

        # Process each prompt in the file
        for prompt_key, prompt_data in data.items():
            if not isinstance(prompt_data, dict):
                continue

            # Build full prompt name
            full_name = folder_prefix + prompt_key

            # Check for collisions at label level
            labels = prompt_data.get("labels", {})
            for label_name in labels.keys():
                # Check if this name+label already exists
                if full_name in self._loaded_prompts:
                    if label_name in self._loaded_prompts[full_name]:
                        # Collision detected
                        previous_file = self._loaded_prompts[full_name][label_name]
                        raise PromptConfigurationError(
                            f"Duplicate prompt definition: '{full_name}' with label '{label_name}' "
                            f"found in multiple files:\n"
                            f"  - {previous_file}\n"
                            f"  - {file_path}"
                        )
                    else:
                        # Same name, different label - OK
                        self._loaded_prompts[full_name][label_name] = str(file_path)
                else:
                    # First time seeing this prompt name
                    self._loaded_prompts[full_name] = {label_name: str(file_path)}

    def _load_prompt_version(self, name: str, label: Optional[str], version: Optional[int]) -> PromptVersion:
        """Load a specific prompt version from YAML files.

        This method is called when cache misses or ttl_seconds=-1.
        It searches through the loaded prompts and returns the matching PromptVersion.

        Args:
            name: Prompt name (with optional folder path)
            label: Label to fetch (e.g., "production")
            version: Version number to fetch

        Returns:
            PromptVersion instance

        Raises:
            PromptNotFoundError: If prompt not found
            PromptVersionNotFoundError: If label/version not found
        """
        # Implementation details omitted for brevity
        # This method fetches from the in-memory representation built by _load_all_prompts()
        pass

    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> PromptVersion:
        """Get prompt from YAML files.

        Args:
            name: Prompt name
            version: Numeric version
            label: Version label
            ttl_seconds: Cache TTL in seconds

        Returns:
            PromptVersion instance

        Note:
            YAML backend reads from local filesystem which is reliable and fast.
        """
        # Validate: cannot specify both label and version
        if version is not None and label is not None:
            raise ValueError(
                "Cannot specify both 'version' and 'label'. "
                "Use label for environment-based access (production, staging) "
                "or version for direct numeric access."
            )

        # Determine target
        if label is not None:
            target_label = label
            target_version = None
        elif version is not None:
            target_label = None
            target_version = version
        else:
            # Default to production label
            target_label = "production"
            target_version = None

        cache_key = (name, target_label, target_version)

        # Determine effective TTL
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.DEFAULT_TTL

        # No caching if ttl_seconds == -1
        if effective_ttl == -1:
            return self._load_prompt_version(name, target_label, target_version)

        # Check cache and expiry
        if cache_key in self._cache:
            from time import time
            cached_value, expiry = self._cache[cache_key]
            if expiry is None or time() < expiry:
                return cached_value
            else:
                # Expired, remove
                del self._cache[cache_key]

        # Load from file
        prompt_version = self._load_prompt_version(name, target_label, target_version)

        # Calculate expiry
        if effective_ttl == 0:
            expiry = None  # Infinite cache
        else:
            from time import time
            expiry = time() + effective_ttl

        # Simple capacity check with FIFO eviction
        # (cachetools.TTLCache would be better for production)
        if len(self._cache) >= self.cache_maxsize:
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        # Store in cache
        self._cache[cache_key] = (prompt_version, expiry)

        return prompt_version
```

### Cache Limitations (YAML Backend)

**File change detection**: The YAML backend cache does NOT automatically detect file changes. If you modify YAML files:
- Option 1: Use `ttl_seconds=-1` to bypass cache
- Option 2: Create a new manager instance
- Option 3: Restart the application

**Tradeoff with `ttl_seconds=0` (infinite cache)**:
- Pro: Maximum performance, no repeated file reads
- Con: File changes require restart
- Best for: Production environments with immutable prompts

**For development**: Use short TTL (5-10 seconds) or `ttl_seconds=-1` for always-fresh prompts.

### Langfuse Backend Caching

Langfuse SDK handles caching internally:
- **Default TTL**: 300 seconds (5 minutes)
- **Cache control**: Via `cache_ttl_seconds` parameter (mapped from `ttl_seconds`)
- **Cache invalidation**: Managed by Langfuse SDK automatically
- **No file change issues**: Prompts fetched from server with automatic updates

### Usage Examples

```python
# Default cache (300 seconds / 5 minutes)
prompt = pm.get_prompt("greeting")

# No caching (always fresh)
prompt = pm.get_prompt("realtime-status", ttl_seconds=-1)

# Infinite cache (never expires)
prompt = pm.get_prompt("legal-disclaimer", ttl_seconds=0)

# Short cache (10 seconds)
prompt = pm.get_prompt("dynamic-banner", ttl_seconds=10)

# Long cache (1 hour)
prompt = pm.get_prompt("terms-of-service", ttl_seconds=3600)

# With label and custom TTL
prompt = pm.get_prompt("greeting", label="staging", ttl_seconds=60)
```

---

## Configuration

### Prompts Directory

**Priority order:**
1. Explicit `prompts_dir` parameter
2. `GRAFLOW_PROMPTS_DIR` environment variable
3. Default: `./prompts`

### YAMLPromptManager Configuration

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

class YAMLPromptManager(PromptManager):
    def __init__(
        self,
        prompts_dir: Optional[str] = None,
        cache_maxsize: int = 1000
    ):
        """Initialize YAML prompt manager.

        Args:
            prompts_dir: Directory containing YAML prompt files
                        If None, uses GRAFLOW_PROMPTS_DIR env var
                        If env var not set, defaults to "./prompts"
            cache_maxsize: Maximum cache entries (default: 1000)
        """
        import os
        from pathlib import Path

        # Resolve prompts directory
        if prompts_dir is None:
            prompts_dir = os.getenv("GRAFLOW_PROMPTS_DIR", "./prompts")

        self.prompts_dir = Path(prompts_dir).resolve()
        self.cache_maxsize = cache_maxsize

        # Validate directory exists
        if not self.prompts_dir.exists():
            raise ValueError(f"Prompts directory not found: {self.prompts_dir}")
```

### Usage

```python
# Default (uses ./prompts or GRAFLOW_PROMPTS_DIR)
pm = PromptManagerFactory.create("yaml")

# Explicit path
pm = PromptManagerFactory.create("yaml", prompts_dir="./custom-prompts")

# Via environment variable
# export GRAFLOW_PROMPTS_DIR=/var/app/prompts
pm = PromptManagerFactory.create("yaml")
```

### PromptManagerFactory

```python
from __future__ import annotations

from typing import Dict, Type, List, Any

class PromptManagerFactory:
    """Factory for creating prompt manager instances."""

    _backends: Dict[str, Type[PromptManager]] = {}

    @classmethod
    def create(
        cls,
        backend: str = "yaml",
        **kwargs
    ) -> PromptManager:
        """Create prompt manager instance.

        Args:
            backend: Backend type ("yaml" or "langfuse")
            **kwargs: Backend-specific configuration

        YAML backend kwargs:
            prompts_dir: Directory path (optional, defaults to GRAFLOW_PROMPTS_DIR or "./prompts")
            cache_maxsize: Max cache entries (default: 1000)

        Langfuse backend kwargs:
            public_key: Langfuse public key (optional, from LANGFUSE_PUBLIC_KEY env)
            secret_key: Langfuse secret key (optional, from LANGFUSE_SECRET_KEY env)
            host: Langfuse host (optional, from LANGFUSE_HOST env)
            fetch_timeout_seconds: Timeout for fetching prompts (optional, uses SDK default if None)
            max_retries: Maximum retry attempts (optional, uses SDK default if None)

        Note:
            Langfuse backend uses SDK's built-in caching (cache_ttl_seconds parameter).
            No custom cache configuration needed.

        Returns:
            PromptManager instance

        Raises:
            ValueError: If backend is unknown

        Example:
            ```python
            # YAML backend
            pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

            # Langfuse backend
            pm = PromptManagerFactory.create("langfuse")
            ```
        """
        if backend not in cls._backends:
            raise ValueError(f"Unknown backend: {backend}")

        manager_class = cls._backends[backend]
        return manager_class(**kwargs)

    @classmethod
    def register_backend(
        cls,
        name: str,
        manager_class: Type[PromptManager]
    ) -> None:
        """Register custom backend."""
        cls._backends[name] = manager_class

    @classmethod
    def get_available_backends(cls) -> List[str]:
        """List available backends."""
        return list(cls._backends.keys())
```

---

## Langfuse Compatibility

### Naming Conventions

Both backends use the same naming:
- **Virtual folders**: Slash-based (e.g., `customer/greeting/welcome`)
- **Labels**: String tags (e.g., `production`, `staging`)
- **Versions**: Numeric IDs (e.g., 1, 2, 3)

### API Compatibility

```python
# Same API for both backends
pm_yaml = PromptManagerFactory.create("yaml", prompts_dir="./prompts")
pm_langfuse = PromptManagerFactory.create("langfuse")

# Same access pattern
prompt1 = pm_yaml.get_prompt("customer/greeting", label="production")
prompt2 = pm_langfuse.get_prompt("customer/greeting", label="production")

# Same rendering
result1 = prompt1.render(name="Alice")
result2 = prompt2.render(name="Alice")
```

### LangfusePromptManager Implementation

```python
from __future__ import annotations

from typing import Optional, Any

class LangfusePromptManager(PromptManager):
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        fetch_timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """Initialize Langfuse prompt manager.

        Args:
            public_key: Langfuse public key (from LANGFUSE_PUBLIC_KEY env if None)
            secret_key: Langfuse secret key (from LANGFUSE_SECRET_KEY env if None)
            host: Langfuse host (from LANGFUSE_HOST env if None)
            fetch_timeout_seconds: Timeout in seconds for fetching prompts from Langfuse server
                                  None = use Langfuse SDK default
            max_retries: Maximum number of retry attempts for failed fetches
                        None = use Langfuse SDK default

        Note:
            Langfuse SDK handles caching internally via cache_ttl_seconds parameter.
            No custom cache implementation needed in this manager.

            fetch_timeout_seconds and max_retries are applied to all get_prompt() calls.
        """
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        self.fetch_timeout_seconds = fetch_timeout_seconds
        self.max_retries = max_retries

    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> PromptVersion:
        """Get prompt from Langfuse.

        Args:
            name: Prompt name
            version: Numeric version
            label: Version label
            ttl_seconds: Cache TTL in seconds
                        None = use Langfuse default (300 seconds)
                        -1 = no caching (cache_ttl_seconds=0)
                        0 = infinite cache (not supported by Langfuse, uses default)
                        >0 = cache for N seconds

        Returns:
            PromptVersion object

        Note:
            Langfuse SDK returns Union[TextPromptClient, ChatPromptClient].
            We convert both to our unified PromptVersion model.

            Langfuse SDK handles both label and version parameters. If both are
            specified, Langfuse applies its own resolution logic (typically label wins).
            For consistency with YAML backend, we raise ValueError for ambiguous requests.

            Caching is handled by Langfuse SDK via cache_ttl_seconds parameter.
            Special value mapping:
            - ttl_seconds=-1 → cache_ttl_seconds=0 (no cache)
            - ttl_seconds=0 → cache_ttl_seconds=None (use default 300s)
            - ttl_seconds>0 → cache_ttl_seconds=ttl_seconds (custom TTL)

            Fetch timeout and max retries are configured at manager initialization.
        """
        # Validate: cannot specify both label and version
        if version is not None and label is not None:
            raise ValueError(
                "Cannot specify both 'version' and 'label'. "
                "Use label for environment-based access or version for direct access."
            )

        # Map ttl_seconds to Langfuse's cache_ttl_seconds
        if ttl_seconds is None:
            cache_ttl = None  # Langfuse default (300 seconds)
        elif ttl_seconds == -1:
            cache_ttl = 0  # No cache
        elif ttl_seconds == 0:
            # Langfuse doesn't support infinite cache, use default
            cache_ttl = None
        else:
            cache_ttl = ttl_seconds

        # Fetch from Langfuse with SDK's built-in caching
        # Returns Union[TextPromptClient, ChatPromptClient]
        langfuse_prompt = self._client.get_prompt(
            name,
            version=version,
            label=label,
            cache_ttl_seconds=cache_ttl,
            max_retries=self.max_retries,
            fetch_timeout_seconds=self.fetch_timeout_seconds,
        )

        # Convert to PromptVersion
        prompt_version = self._convert_langfuse_prompt(langfuse_prompt)

        return prompt_version

    def _convert_langfuse_prompt(
        self,
        langfuse_prompt: Union[Any, Any]  # TextPromptClient | ChatPromptClient
    ) -> PromptVersion:
        """Convert Langfuse prompt object to PromptVersion.

        Args:
            langfuse_prompt: Langfuse prompt object (TextPromptClient or ChatPromptClient)

        Returns:
            PromptVersion instance

        Note:
            Langfuse SDK returns different types:
            - TextPromptClient: has .prompt (str) and .type == 'text'
            - ChatPromptClient: has .prompt (List[ChatMessageDict]) and .type == 'chat'

            Both have: .name, .version, .labels, .config
        """
        # Detect type from the prompt object
        # Langfuse SDK provides 'type' attribute
        if hasattr(langfuse_prompt, 'type'):
            prompt_type = langfuse_prompt.type  # 'text' or 'chat'
        else:
            # Fallback: infer from content type
            prompt_type = 'chat' if isinstance(langfuse_prompt.prompt, list) else 'text'

        # Extract content (already in the right format)
        content = langfuse_prompt.prompt

        # Extract version info
        version_num = getattr(langfuse_prompt, 'version', None)

        # Extract label (Langfuse stores as list, take first)
        labels = getattr(langfuse_prompt, 'labels', [])
        label = labels[0] if labels else None

        return PromptVersion(
            name=langfuse_prompt.name,
            content=content,
            type=prompt_type,
            version=version_num,
            label=label,
            created_at=None,  # Langfuse doesn't expose created_at in client objects
            metadata=getattr(langfuse_prompt, 'config', {})  # Store config as metadata
        )
```

### Migration Path (YAML → Langfuse)

Prompts can be easily migrated from YAML to Langfuse:

```python
# Export from YAML
yaml_pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

# Import to Langfuse (manual or via script)
for prompt_name in yaml_pm.list_prompts():
    for label in yaml_pm.list_labels(prompt_name):
        prompt = yaml_pm.get_prompt(prompt_name, label=label, ttl_seconds=-1)

        # Create in Langfuse via UI or API
        # (Langfuse Python SDK doesn't support prompt creation yet)
```

---

## Integration with Graflow

### WorkflowContext Integration (Primary Pattern)

**Modifications to graflow/core/workflow.py:**

The `workflow()` function accepts an optional `prompt_manager` parameter:

```python
from __future__ import annotations

from typing import Optional

def workflow(
    name: str,
    tracer: Optional[Tracer] = None,
    prompt_manager: Optional[PromptManager] = None  # NEW
) -> WorkflowContext:
    """Context manager for creating a workflow.

    Args:
        name: Name of the workflow
        tracer: Optional tracer for workflow execution tracking
        prompt_manager: Optional PromptManager instance for prompt management

    Returns:
        WorkflowContext instance

    Example:
        ```python
        from graflow.core.workflow import workflow
        from graflow.prompts import PromptManagerFactory

        # Create prompt manager
        pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

        # Pass to workflow
        with workflow("customer_engagement", prompt_manager=pm) as wf:
            # Access via wf.prompt_manager
            task1.run()
            task2.run()
        ```
    """
    return WorkflowContext(name, tracer=tracer, prompt_manager=prompt_manager)
```

**Update WorkflowContext.__init__():**

```python
from __future__ import annotations

from typing import Optional, Any

class WorkflowContext:
    def __init__(
        self,
        name: str,
        tracer: Optional[Tracer] = None,
        prompt_manager: Optional[PromptManager] = None  # NEW
    ):
        """Initialize a new workflow context.

        Args:
            name: Name for this workflow
            tracer: Optional tracer for workflow execution tracking
            prompt_manager: Optional PromptManager instance for prompt management
        """
        self.name = name
        self.graph = TaskGraph()
        self._task_counter = 0
        self._group_counter = 0
        self._redis_client: Optional[Any] = None
        self._tracer = tracer
        self._llm_agent_providers: dict[str, LLMAgentProvider] = {}
        self._token: Optional[contextvars.Token] = None
        self._prompt_manager: Optional[PromptManager] = prompt_manager  # NEW

    @property
    def prompt_manager(self) -> Optional[PromptManager]:
        """Get prompt manager for this workflow.

        Returns:
            PromptManager instance if set, None otherwise

        Example:
            ```python
            pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

            with workflow("customer_engagement", prompt_manager=pm) as wf:
                # Access prompt manager
                if wf.prompt_manager:
                    prompt = wf.prompt_manager.get_prompt("greeting")
            ```
        """
        return self._prompt_manager
```

**Update WorkflowContext.execute():**

Pass `prompt_manager` to `ExecutionContext.create()`:

```python
    def execute(
        self,
        start_node: Optional[str] = None,
        max_steps: int = 10000,
        ret_context: bool = False,
        initial_channel: Optional[dict[str, Any]] = None,
    ) -> Any | tuple[Any, ExecutionContext]:
        """Execute the workflow starting from the specified node."""

        # ... existing start_node logic ...

        from graflow.core.context import ExecutionContext
        from graflow.core.engine import WorkflowEngine

        exec_context = ExecutionContext.create(
            self.graph,
            start_node,
            max_steps=max_steps,
            tracer=self._tracer,
            prompt_manager=self._prompt_manager  # NEW: Pass to execution context
        )

        # ... rest of execute logic ...
```

---

### ExecutionContext Integration (Secondary Pattern)

**Modifications to graflow/core/context.py:**

**Add to ExecutionContext.__init__():**

```python
from __future__ import annotations

from typing import Optional, Dict, Any

class ExecutionContext:
    def __init__(
        self,
        graph: TaskGraph,
        start_node: Optional[str] = None,
        # ... existing parameters ...
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,  # NEW
    ):
        # ... existing initialization ...

        # LLM integration (existing)
        self._llm_client: Optional[LLMClient] = llm_client
        self._llm_agents: Dict[str, Any] = {}
        self._llm_agents_yaml: Dict[str, str] = {}

        # Prompt management (NEW)
        self._prompt_manager: Optional[PromptManager] = prompt_manager

        # HITL integration (existing)
        # ...
```

**Add prompt_manager property:**

```python
    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager instance.

        Lazily creates a default YAMLPromptManager if not explicitly set.
        Prompts directory is resolved from GRAFLOW_PROMPTS_DIR environment variable,
        falling back to "./prompts".

        Returns:
            PromptManager instance

        Example:
            ```python
            # .env file:
            # GRAFLOW_PROMPTS_DIR=./prompts

            # Access prompt manager (auto-created with YAML backend)
            pm = context.prompt_manager
            prompt = pm.get_prompt("greeting", label="production")
            result = prompt.render(name="Alice")

            # Or inject explicitly
            from graflow.prompts import PromptManagerFactory
            pm = PromptManagerFactory.create("yaml", prompts_dir="./my-prompts")
            context = ExecutionContext.create(
                graph, start_node,
                prompt_manager=pm
            )
            ```
        """
        if self._prompt_manager is None:
            # Lazy initialization: create default YAML prompt manager
            from graflow.prompts.factory import PromptManagerFactory

            # Default to YAML backend with environment-based prompts_dir
            self._prompt_manager = PromptManagerFactory.create("yaml")

        return self._prompt_manager
```

**Update ExecutionContext.create() classmethod:**

```python
    @classmethod
    def create(
        cls,
        graph: TaskGraph,
        start_node: Optional[str] = None,
        max_steps: int = 10000,
        default_max_cycles: int = 10,
        default_max_retries: int = 3,
        channel_backend: str = "memory",
        config: Optional[Dict[str, Any]] = None,
        tracer: Optional[Tracer] = None,
        llm_client: Optional[LLMClient] = None,
        prompt_manager: Optional[PromptManager] = None,  # NEW
    ) -> ExecutionContext:
        """Create a new execution context.

        Args:
            graph: Task graph defining the workflow
            start_node: Optional starting task node
            max_steps: Maximum execution steps
            default_max_cycles: Default maximum cycles for tasks
            default_max_retries: Default maximum retry attempts
            channel_backend: Backend for inter-task communication (default: memory)
            config: Configuration for queue and channel
            tracer: Optional tracer for workflow execution tracking
            llm_client: Optional LLMClient instance for LLM integration
            prompt_manager: Optional PromptManager instance for prompt management

        Example:
            ```python
            from graflow.prompts import PromptManagerFactory

            # Create prompt manager
            pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

            # Create context with prompt manager
            context = ExecutionContext.create(
                graph, start_node,
                prompt_manager=pm
            )

            # Or let it auto-initialize from GRAFLOW_PROMPTS_DIR
            context = ExecutionContext.create(graph, start_node)
            prompt = context.prompt_manager.get_prompt("greeting")
            ```
        """
        if start_node is None:
            candidate_nodes = graph.get_start_nodes()
            if candidate_nodes:
                start_node = candidate_nodes[0]

        return cls(
            graph=graph,
            start_node=start_node,
            max_steps=max_steps,
            default_max_cycles=default_max_cycles,
            default_max_retries=default_max_retries,
            channel_backend=channel_backend,
            config=config,
            tracer=tracer,
            llm_client=llm_client,
            prompt_manager=prompt_manager,  # NEW
        )
```

**Add to __getstate__() for serialization:**

```python
    def __getstate__(self) -> dict:
        """Pickle serialization: exclude un-serializable objects."""
        state = self.__dict__.copy()

        # ... existing serialization ...

        # Prompt manager: Keep configuration but not the instance
        # Will be reconstructed in __setstate__
        state.pop("_prompt_manager", None)

        return state
```

**Add to __setstate__() for deserialization:**

```python
    def __setstate__(self, state: dict) -> None:
        """Pickle deserialization: reconstruct excluded objects."""
        self.__dict__.update(state)

        # ... existing reconstruction ...

        # Ensure prompt manager attribute exists for older checkpoints
        if not hasattr(self, "_prompt_manager"):
            self._prompt_manager = None
```

### TaskExecutionContext Integration

**Add property to TaskExecutionContext:**

```python
from __future__ import annotations

class TaskExecutionContext:
    # ... existing code ...

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager from execution context.

        Returns:
            PromptManager instance (auto-created if not set)

        Example:
            ```python
            @task(inject_context=True)
            def my_task(context: TaskExecutionContext):
                # Access prompt manager through task context
                prompt = context.prompt_manager.get_prompt(
                    "greeting",
                    label="production"
                )
                result = prompt.render(name="Alice")
                return result
            ```
        """
        return self.execution_context.prompt_manager
```

### Usage in Tasks

**Option 1: Via ExecutionContext (inject_context=True):**

```python
from graflow.core.decorators import task
from graflow.core.context import ExecutionContext

@task(inject_context=True)
def send_welcome_email(context: ExecutionContext, user_name: str, user_email: str):
    # Access prompt manager from context
    prompt = context.prompt_manager.get_prompt(
        "customer/onboarding/welcome-email",
        label="production"
    )

    # Render with variables
    email_body = prompt.render(
        name=user_name,
        product="Graflow",
        support_email="support@graflow.com"
    )

    # Send email
    send_email(user_email, "Welcome!", email_body)

    return {"status": "sent", "to": user_email}
```

**Option 2: Via TaskExecutionContext (also inject_context=True):**

```python
from graflow.core.decorators import task
from graflow.core.context import TaskExecutionContext

@task(inject_context=True)
def send_notification(context: TaskExecutionContext, user_id: str, event: str):
    # Access prompt manager through task context (cleaner for task-level code)
    prompt = context.prompt_manager.get_prompt(
        f"notifications/{event}",
        label="production"
    )

    # Render notification
    notification = prompt.render(user_id=user_id, timestamp=datetime.now())

    # Send notification
    send_push_notification(user_id, notification)

    return {"user_id": user_id, "event": event, "sent": True}
```

### Workflow Configuration

**Option 1: Explicit prompt manager (YAML):**

```python
from graflow.core.workflow import workflow
from graflow.prompts import PromptManagerFactory

# Create YAML prompt manager
pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

# Pass to workflow
with workflow("onboarding_workflow", prompt_manager=pm) as wf:
    send_welcome_email.run(user_name="Alice", user_email="alice@example.com")
```

**Option 2: Environment-based (lazy initialization):**

```bash
# .env file
export GRAFLOW_PROMPTS_DIR=./prompts
```

```python
from graflow.core.workflow import workflow

# Workflow without explicit prompt manager
with workflow("onboarding_workflow") as wf:
    # Prompt manager auto-initialized on first access from GRAFLOW_PROMPTS_DIR
    send_welcome_email.run(user_name="Alice", user_email="alice@example.com")
```

**Option 3: Langfuse backend for production:**

```bash
# .env file
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com
```

```python
from graflow.core.workflow import workflow
from graflow.prompts import PromptManagerFactory

# Create Langfuse manager (credentials from env vars)
pm = PromptManagerFactory.create(
    "langfuse",
    fetch_timeout_seconds=10,  # 10 second timeout for all fetches
    max_retries=2              # Retry up to 2 times if fetch fails
)

# Pass to workflow
with workflow("production_workflow", prompt_manager=pm) as wf:
    send_welcome_email.run(user_name="Alice", user_email="alice@example.com")
```

---

## Usage Examples

### Example 1: Basic YAML Usage

```python
from graflow.prompts import PromptManagerFactory

# Create YAML manager
pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

# Get and render text prompt
prompt = pm.get_prompt("greeting", label="production")
result = prompt.render(name="Alice", product="Graflow")
print(result)  # "Hello Alice, welcome to Graflow!"

# Get and render chat prompt
prompt = pm.get_prompt("interview", label="production")
messages = prompt.render(domain="AI", topic="transformers")
print(messages)
# [
#   {'role': 'system', 'content': 'You are an expert in AI.'},
#   {'role': 'user', 'content': 'Interview me about transformers.'}
# ]
```

---

### Example 2: Langfuse Backend

```python
from graflow.prompts import PromptManagerFactory

# Create Langfuse manager with timeout and retry configuration
pm = PromptManagerFactory.create(
    "langfuse",
    fetch_timeout_seconds=10,
    max_retries=3  # Retry up to 3 times if fetch fails
)

# Same API as YAML
prompt = pm.get_prompt("customer/greeting", label="production")
result = prompt.render(name="Bob", product="Graflow")
```

---

### Example 3: Version Management

```python
pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

# Production version
prod_prompt = pm.get_prompt("greeting", label="production")
prod_result = prod_prompt.render(name="Alice")

# Staging version (for testing)
staging_prompt = pm.get_prompt("greeting", label="staging")
staging_result = staging_prompt.render(name="Alice")

# Specific version by number
v3_prompt = pm.get_prompt("greeting", version=3)
v3_result = v3_prompt.render(name="Alice")

# Check version info
print(prod_prompt.version)  # 3
print(prod_prompt.label)    # "production"
print(staging_prompt.version)  # 4
```

---

### Example 4: Cache Control

```python
pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

# Default cache (5 minutes)
prompt = pm.get_prompt("greeting")

# No cache (always fresh)
prompt = pm.get_prompt("realtime-banner", ttl_seconds=-1)

# Infinite cache
prompt = pm.get_prompt("legal-terms", ttl_seconds=0)

# Custom short cache (10 seconds)
prompt = pm.get_prompt("limited-offer", ttl_seconds=10)

# Custom long cache (1 hour)
prompt = pm.get_prompt("faq-answer", ttl_seconds=3600)
```

---

### Example 5: Workflow Integration

```python
from graflow.core.workflow import workflow
from graflow.core.decorators import task
from graflow.core.context import ExecutionContext
from graflow.prompts import PromptManagerFactory

@task(inject_context=True)
def greet_customer(context: ExecutionContext, customer_name: str, tier: str):
    # Select prompt based on customer tier
    if tier == "vip":
        prompt_name = "customer/greeting/vip"
    else:
        prompt_name = "customer/greeting/standard"

    # Get and render prompt
    prompt = context.prompt_manager.get_prompt(prompt_name, label="production")
    greeting = prompt.render(
        customer_name=customer_name,
        product="Graflow"
    )

    return greeting

@task(inject_context=True)
def send_interview_request(context: ExecutionContext, expert_name: str, domain: str):
    # Get chat prompt
    prompt = context.prompt_manager.get_prompt(
        "interviews/expert-request",
        label="production"
    )

    # Render to messages
    messages = prompt.render(
        expert_name=expert_name,
        domain=domain,
        company="Graflow Inc."
    )

    # Use with LLM
    from graflow.llm import LLMClient
    llm = LLMClient()
    response = llm.completion(messages=messages)

    return response

# Create prompt manager
pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

# Pass to workflow
with workflow("customer_engagement", prompt_manager=pm) as wf:
    # Run tasks
    greet_customer.run(customer_name="Alice", tier="vip")
    send_interview_request.run(expert_name="Dr. Smith", domain="machine learning")
```

---

### Example 6: A/B Testing

```python
import random
from graflow.core.decorators import task

@task(inject_context=True)
def ab_test_greeting(context, user_name: str):
    # Randomly select variant
    variant = random.choice(["production", "experiment-a", "experiment-b"])

    # Get prompt for variant
    prompt = context.prompt_manager.get_prompt(
        "customer/greeting",
        label=variant
    )

    greeting = prompt.render(name=user_name, product="Graflow")

    return {
        "variant": variant,
        "greeting": greeting,
        "user": user_name
    }
```

---

### Example 7: Langfuse with Retries (Production Resilience)

```python
from graflow.core.workflow import workflow
from graflow.core.decorators import task
from graflow.prompts import PromptManagerFactory

# Create Langfuse manager with timeout and retry configuration
pm = PromptManagerFactory.create(
    "langfuse",
    fetch_timeout_seconds=5,
    max_retries=3  # Retry up to 3 times if fetch fails
)

@task(inject_context=True)
def send_critical_notification(context, user_id: str, event: str):
    # Fetch prompt (uses manager-level retry configuration)
    prompt = context.prompt_manager.get_prompt(
        "notifications/critical",
        label="production",
    )

    # Render notification
    message = prompt.render(user_id=user_id, event=event)

    # Send notification
    send_push_notification(user_id, message)

    return {"status": "sent", "user_id": user_id}

with workflow("critical_alerts", prompt_manager=pm) as wf:
    send_critical_notification.run(user_id="user123", event="system_down")
```

---

## Custom Exceptions

```python
# graflow/prompts/exceptions.py
from __future__ import annotations

class PromptError(Exception):
    """Base exception for prompt management errors."""
    pass

class PromptNotFoundError(PromptError):
    """Raised when prompt is not found."""
    pass

class PromptVersionNotFoundError(PromptError):
    """Raised when prompt version/label is not found."""
    pass

class PromptCompilationError(PromptError):
    """Raised when template rendering fails."""
    pass

class PromptConfigurationError(PromptError):
    """Raised when prompt manager configuration is invalid or duplicate prompts detected."""
    pass
```

**Usage Examples:**

```python
from graflow.prompts import (
    PromptNotFoundError,
    PromptVersionNotFoundError,
    PromptCompilationError,
    PromptConfigurationError
)

# Handle missing prompt
try:
    prompt = pm.get_prompt("nonexistent-prompt")
except PromptNotFoundError as e:
    print(f"Prompt not found: {e}")

# Handle missing variable
try:
    prompt = pm.get_prompt("greeting")
    result = prompt.render()  # Missing 'name' variable
except PromptCompilationError as e:
    print(f"Failed to render: {e}")

# Handle duplicate prompt definition
try:
    pm = YAMLPromptManager(prompts_dir="./prompts")
except PromptConfigurationError as e:
    # Raised if two files define same prompt name + label
    print(f"Configuration error: {e}")
    # Example: "Duplicate prompt definition: 'customer/greeting' with label 'production' found in multiple files"
```

---

## Testing Strategy

### Unit Tests

```python
# tests/prompts/test_yaml_manager.py

import pytest
from graflow.prompts import YAMLPromptManager, PromptNotFoundError

def test_get_prompt_basic(tmp_path):
    # Create test prompt file
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    (prompts_dir / "test.yaml").write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}!"
      version: 1
""")

    # Create manager
    pm = YAMLPromptManager(prompts_dir=str(prompts_dir))

    # Get prompt
    prompt = pm.get_prompt("greeting", label="production")
    assert prompt.name == "greeting"
    assert prompt.type == "text"
    assert prompt.version == 1

    # Render
    result = prompt.render(name="Alice")
    assert result == "Hello Alice!"

def test_virtual_folders(tmp_path):
    # Create nested structure
    prompts_dir = tmp_path / "prompts"
    customer_dir = prompts_dir / "customer"
    customer_dir.mkdir(parents=True)

    (customer_dir / "prompts.yaml").write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}!"
      version: 1
""")

    pm = YAMLPromptManager(prompts_dir=str(prompts_dir))

    # Access with virtual folder
    prompt = pm.get_prompt("customer/greeting")
    result = prompt.render(name="Bob")
    assert result == "Hello Bob!"

def test_cache_ttl(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    (prompts_dir / "test.yaml").write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}!"
      version: 1
""")

    pm = YAMLPromptManager(prompts_dir=str(prompts_dir))

    # First call (cache miss)
    prompt1 = pm.get_prompt("greeting")

    # Second call (cache hit)
    prompt2 = pm.get_prompt("greeting")

    # Should be same object (cached)
    assert prompt1 is prompt2

    # No cache
    prompt3 = pm.get_prompt("greeting", ttl_seconds=-1)
    # New object loaded
    assert prompt1 is not prompt3

def test_chat_prompt(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    (prompts_dir / "test.yaml").write_text("""
interview:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are an expert in {{domain}}."
        - role: user
          content: "Interview me about {{topic}}."
      version: 1
""")

    pm = YAMLPromptManager(prompts_dir=str(prompts_dir))
    prompt = pm.get_prompt("interview")

    messages = prompt.render(domain="AI", topic="GPT")

    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert messages[0]['content'] == 'You are an expert in AI.'
    assert messages[1]['role'] == 'user'
    assert messages[1]['content'] == 'Interview me about GPT.'

def test_collision_detection(tmp_path):
    """Test that duplicate prompt name+label raises PromptConfigurationError."""
    from graflow.prompts import YAMLPromptManager, PromptConfigurationError

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create two files with same prompt name and label
    (prompts_dir / "prompt1.yaml").write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello from file 1"
      version: 1
""")

    (prompts_dir / "prompt2.yaml").write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello from file 2"
      version: 1
""")

    # Should raise PromptConfigurationError
    with pytest.raises(PromptConfigurationError) as exc_info:
        YAMLPromptManager(prompts_dir=str(prompts_dir))

    assert "Duplicate prompt definition" in str(exc_info.value)
    assert "greeting" in str(exc_info.value)
    assert "production" in str(exc_info.value)

def test_no_collision_different_labels(tmp_path):
    """Test that same prompt name with different labels is allowed."""
    from graflow.prompts import YAMLPromptManager

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create two files with same prompt name but different labels
    (prompts_dir / "prompt1.yaml").write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello production"
      version: 1
""")

    (prompts_dir / "prompt2.yaml").write_text("""
greeting:
  type: text
  labels:
    staging:
      content: "Hello staging"
      version: 1
""")

    # Should NOT raise - different labels are OK
    pm = YAMLPromptManager(prompts_dir=str(prompts_dir))

    # Verify both labels accessible
    prod_prompt = pm.get_prompt("greeting", label="production")
    assert "production" in prod_prompt.render()

    staging_prompt = pm.get_prompt("greeting", label="staging")
    assert "staging" in staging_prompt.render()
```

### Integration Tests

```python
# tests/prompts/test_prompt_integration.py

import pytest
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.decorators import task

@pytest.mark.integration
def test_prompt_in_workflow(tmp_path):
    # Setup prompts
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    (prompts_dir / "test.yaml").write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}!"
      version: 1
""")

    # Create context with prompt manager
    graph = TaskGraph()
    context = ExecutionContext.create(
        graph,
        prompt_backend="yaml",
        prompt_config={"prompts_dir": str(prompts_dir)}
    )

    # Define task
    @task(inject_context=True)
    def greet(context, name: str):
        prompt = context.prompt_manager.get_prompt("greeting")
        return prompt.render(name=name)

    # Execute
    result = greet.run(name="Alice")
    assert result == "Hello Alice!"
```

---

## Dependencies

### Required Dependencies

All required dependencies are already in `pyproject.toml`:

```toml
[project]
dependencies = [
    "jinja2>=3.1.6",      # Already present - Template rendering
    "pyyaml>=6.0",        # Already present - YAML parsing
    "cachetools>=5.3.2",  # Already present - Caching
]
```

### Optional Dependencies

```toml
[project.optional-dependencies]
tracing = [
    "langfuse>=3.8.1",  # Already present - Langfuse backend
]
```

### Installation

```bash
# Basic (YAML backend only)
uv pip install graflow

# With Langfuse support
uv pip install graflow[tracing]

# All features
uv pip install graflow[all]
```

---

## Environment Variables

### YAML Backend

```bash
# Optional: Override default prompts directory
export GRAFLOW_PROMPTS_DIR=./prompts
```

### Langfuse Backend

```bash
# Required for Langfuse backend
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

# Optional: Custom Langfuse host
export LANGFUSE_HOST=https://cloud.langfuse.com
```

---

## Implementation Roadmap

### Phase 1: Core Structure
1. Create `graflow/prompts/` directory
2. Implement `exceptions.py` (custom exceptions)
3. Implement `models.py` (Prompt, PromptVersion)
4. Implement `base.py` (PromptManager abstract class)

### Phase 2: YAML Backend
1. Implement `yaml_manager.py`
   - `_load_all_prompts()` - Recursively scan directories with collision detection
   - `_load_prompt_file()` - Parse YAML to Prompt objects, check for duplicates
   - `_load_prompt_version()` - Load specific version from in-memory prompts
   - `get_prompt()` - Fetch with caching
   - `list_prompts()`, `list_labels()` - Listing methods
2. Add collision detection tracking (`_loaded_prompts` dict)
3. Add cache implementation with TTL support
4. Test with various directory structures
5. Test collision detection (same name+label in multiple files)

### Phase 3: Langfuse Backend
1. Implement `langfuse_manager.py`
   - Optional import with `LANGFUSE_AVAILABLE` flag
   - `get_prompt()` - Fetch from Langfuse (delegates caching to SDK via `cache_ttl_seconds`)
   - Credential loading from env vars
   - Map `ttl_seconds` to Langfuse's `cache_ttl_seconds` parameter
2. Implement `_convert_langfuse_prompt()` to convert Langfuse prompt objects to PromptVersion
3. No custom cache implementation needed (Langfuse SDK handles it)

### Phase 4: Factory & Public API
1. Implement `factory.py` (PromptManagerFactory)
   - Backend registration
   - `create()` method
   - Conditional Langfuse registration
2. Implement `__init__.py` (public API exports)

### Phase 5: Graflow Integration
1. Modify `graflow/core/workflow.py`
   - Add `prompt_manager` parameter to `workflow()` function
   - Add `prompt_manager` parameter to `WorkflowContext.__init__()`
   - Add `prompt_manager` property to `WorkflowContext`
   - Pass `prompt_manager` to `ExecutionContext.create()` in `execute()`
2. Modify `graflow/core/context.py`
   - Add `prompt_manager` parameter to `ExecutionContext.__init__()`
   - Add lazy `prompt_manager` property
   - Update `create()` classmethod
   - Update `__getstate__()` and `__setstate__()` for serialization
   - Add `prompt_manager` property to `TaskExecutionContext`

### Phase 6: Testing
1. Unit tests (`tests/prompts/`)
   - `test_yaml_manager.py`
   - `test_langfuse_manager.py`
   - `test_factory.py`
   - `test_models.py`
2. Integration tests
   - `test_prompt_integration.py` (with ExecutionContext)
3. Create test fixtures (sample YAML files)

### Phase 7: Documentation & Examples
1. Create `examples/prompts/`
   - `basic_yaml_prompts.py`
   - `langfuse_prompts.py`
   - `version_management.py`
   - `ab_testing.py`
2. Update `CLAUDE.md` with prompt management section

---

## Success Criteria

1. ✅ YAML backend loads prompts from filesystem with virtual folders
2. ✅ Langfuse backend fetches prompts from Langfuse server
3. ✅ Factory pattern allows backend selection
4. ✅ ExecutionContext integration works (lazy initialization)
5. ✅ Variable substitution with Jinja2 `{{variable}}` syntax
6. ✅ Label and version system functions correctly
7. ✅ Per-call TTL caching works as specified
8. ✅ Prompts directory configurable with env var fallback
9. ✅ Error handling covers all edge cases
10. ✅ Type hints pass mypy check
11. ✅ All tests pass
12. ✅ Examples run successfully

---

## Summary

This design provides:

1. **Unified template syntax**: `{{variable}}` works across both backends
2. **Langfuse API compatibility**:
   - Support for max_retries and fetch_timeout_seconds (manager-level configuration)
   - Unified PromptVersion abstraction over TextPromptClient/ChatPromptClient
3. **Flexible organization**: Virtual folders map to filesystem directories
4. **Version control**: Labels for environments, numeric versions for tracking
5. **Smart caching**:
   - YAML: Custom cache with per-call TTL control
   - Langfuse: SDK's built-in caching via `cache_ttl_seconds`
6. **Easy configuration**:
   - YAML: Optional prompts_dir with env var fallback
   - Langfuse: fetch_timeout_seconds and max_retries configured at manager initialization
7. **Workflow integration**: Direct prompt_manager parameter to `workflow()`
8. **Collision detection**: Prevents duplicate prompt definitions across files (YAML)
9. **Production-ready**: Error handling, retry support, type safety, and optimized performance

The module enables teams to:
- Develop locally with YAML files
- Deploy to production with Langfuse
- Version prompts for A/B testing and experiments
- Integrate prompts seamlessly into Graflow workflows via `workflow()` context manager
- Detect configuration errors early (duplicate prompt definitions)
