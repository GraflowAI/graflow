# Prompt Management Module Design

## Overview

Design document for implementing prompt management in Graflow with two backend implementations:
1. **YAMLPromptManager** (default): Local filesystem storage with YAML files
2. **LangFusePromptManager**: LangFuse cloud/server integration for versioned prompt management

### Purpose

- Centralized prompt management for LLM workflows
- Version control for prompts (production, staging, experiments)
- Template variable substitution with Jinja2
- Seamless integration between local development (YAML) and production (LangFuse)
- Support for both single prompts and conversation prompts

### Key Design Decisions

1. **Template Format**: `{{variable}}` (Mustache/Jinja2 syntax)
2. **Dependency**: Jinja2 (already in core dependencies)
3. **Virtual Folders**: Slash-based prompt names (e.g., `customer/greeting/welcome`)
4. **Directory Mapping**: Filesystem directories map to LangFuse-compatible virtual folders
5. **Versioning**: Labels (`production`, `staging`, `latest`) with numeric version tracking
6. **API Pattern**: `get_prompt()` returns `PromptVersion` object, `.render()` compiles template
7. **Caching**: Per-call `ttl_seconds` parameter with smart defaults
8. **Configuration**: `prompts_dir` optional with `GRAFLOW_PROMPTS_DIR` env var fallback

---

## Architecture

### Module Structure

```
graflow/prompts/
├── __init__.py              # Public API exports
├── base.py                  # PromptManager abstract base class
├── models.py                # Data models (Prompt, PromptVersion)
├── yaml_manager.py          # YAML backend implementation (default)
├── langfuse_manager.py      # LangFuse backend implementation
├── factory.py               # PromptManagerFactory
└── exceptions.py            # Prompt-specific exceptions
```

### Class Hierarchy

```
PromptManager (ABC)
├── YAMLPromptManager       # Local YAML files
└── LangFusePromptManager   # LangFuse cloud/server
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

LangFuse uses **virtual folders** via slashes in prompt names (e.g., `customer/greeting/welcome`). Our YAML backend maps filesystem directories to these virtual folders for LangFuse compatibility.

### Mapping Rule

**Full prompt name = `directory_path` + `/` + `yaml_key`**

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
    ) -> "PromptVersion":
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
            - If both specified, label takes precedence
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
from typing import Union, List, Dict, Any, Optional, Literal
from jinja2 import Template

class PromptVersion:
    """Represents a specific version of a prompt."""

    def __init__(
        self,
        name: str,
        content: Union[str, List[Dict[str, str]]],
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

    def render(self, **variables) -> Union[str, List[Dict[str, str]]]:
        """Render template with variables using Jinja2.

        Args:
            **variables: Template variables to substitute

        Returns:
            Compiled prompt (string for 'text', message list for 'chat')

        Raises:
            PromptCompilationError: If template rendering fails

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
        if self.type == 'text':
            template = Template(self.content)
            return template.render(**variables)
        elif self.type == 'chat':
            # Compile each message content
            compiled_messages = []
            for msg in self.content:
                template = Template(msg['content'])
                compiled_messages.append({
                    'role': msg['role'],
                    'content': template.render(**variables)
                })
            return compiled_messages
        else:
            raise ValueError(f"Unknown prompt type: {self.type}")
```

---

## Versioning: Labels vs Versions

### Concepts

- **Version**: Immutable numeric ID (1, 2, 3...) - identifies specific prompt iteration
- **Label**: Mutable string tag ("production", "staging", "latest") - pointer to a version
- **Multiple labels can point to same version**

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

### Cache Implementation

```python
from time import time
from typing import Dict, Tuple, Optional

class YAMLPromptManager(PromptManager):
    DEFAULT_TTL = 300  # 5 minutes

    def __init__(
        self,
        prompts_dir: Optional[str] = None,
        cache_maxsize: int = 1000
    ):
        # Resolve prompts directory
        if prompts_dir is None:
            prompts_dir = os.getenv("GRAFLOW_PROMPTS_DIR", "./prompts")

        self.prompts_dir = Path(prompts_dir).resolve()
        self.cache_maxsize = cache_maxsize

        # Cache: (name, label, version) -> (PromptVersion, expiry_timestamp)
        self._cache: Dict[Tuple, Tuple[PromptVersion, Optional[float]]] = {}

    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> PromptVersion:
        target_label = label or "production"
        cache_key = (name, target_label, version)

        # Determine effective TTL
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.DEFAULT_TTL

        # No caching if ttl_seconds == -1
        if effective_ttl == -1:
            return self._load_prompt_version(name, target_label, version)

        # Check cache and expiry
        if cache_key in self._cache:
            cached_value, expiry = self._cache[cache_key]
            if expiry is None or time() < expiry:
                return cached_value
            else:
                del self._cache[cache_key]

        # Load from file
        prompt_version = self._load_prompt_version(name, target_label, version)

        # Calculate expiry
        if effective_ttl == 0:
            expiry = None  # Infinite
        else:
            expiry = time() + effective_ttl

        # Evict if at capacity (simple FIFO)
        if len(self._cache) >= self.cache_maxsize:
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        # Store in cache
        self._cache[cache_key] = (prompt_version, expiry)

        return prompt_version
```

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

        LangFuse backend kwargs:
            public_key: LangFuse public key (optional, from LANGFUSE_PUBLIC_KEY env)
            secret_key: LangFuse secret key (optional, from LANGFUSE_SECRET_KEY env)
            host: LangFuse host (optional, from LANGFUSE_HOST env)
            cache_maxsize: Max cache entries (default: 1000)

        Returns:
            PromptManager instance

        Raises:
            ValueError: If backend is unknown

        Example:
            ```python
            # YAML backend
            pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

            # LangFuse backend
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

## LangFuse Compatibility

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

### LangFusePromptManager Implementation

```python
class LangFusePromptManager(PromptManager):
    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        cache_maxsize: int = 1000
    ):
        """Initialize LangFuse prompt manager.

        Args:
            public_key: LangFuse public key (from LANGFUSE_PUBLIC_KEY env if None)
            secret_key: LangFuse secret key (from LANGFUSE_SECRET_KEY env if None)
            host: LangFuse host (from LANGFUSE_HOST env if None)
            cache_maxsize: Max cache entries (default: 1000)
        """
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host
        )
        self.cache_maxsize = cache_maxsize
        self._cache: Dict[Tuple, Tuple[PromptVersion, Optional[float]]] = {}

    def get_prompt(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> PromptVersion:
        """Get prompt from LangFuse with caching."""
        cache_key = (name, version, label)
        effective_ttl = ttl_seconds if ttl_seconds is not None else 60  # LangFuse default

        # No caching if ttl_seconds == -1
        if effective_ttl == -1:
            return self._fetch_from_langfuse(name, version, label)

        # Check cache
        if cache_key in self._cache:
            cached_value, expiry = self._cache[cache_key]
            if expiry is None or time() < expiry:
                return cached_value
            else:
                del self._cache[cache_key]

        # Fetch from LangFuse
        langfuse_prompt = self._client.get_prompt(
            name,
            version=version,
            label=label
        )

        # Convert to PromptVersion
        prompt_version = self._convert_langfuse_prompt(langfuse_prompt)

        # Cache it
        if effective_ttl == 0:
            expiry = None
        else:
            expiry = time() + effective_ttl

        self._cache[cache_key] = (prompt_version, expiry)

        return prompt_version
```

### Migration Path (YAML → LangFuse)

Prompts can be easily migrated from YAML to LangFuse:

```python
# Export from YAML
yaml_pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

# Import to LangFuse (manual or via script)
for prompt_name in yaml_pm.list_prompts():
    for label in yaml_pm.list_labels(prompt_name):
        prompt = yaml_pm.get_prompt(prompt_name, label=label, ttl_seconds=-1)

        # Create in LangFuse via UI or API
        # (LangFuse Python SDK doesn't support prompt creation yet)
```

---

## ExecutionContext Integration

### Context Properties

```python
class ExecutionContext:
    def __init__(
        self,
        # ... existing parameters ...
        prompt_manager: Optional[PromptManager] = None,
        prompt_backend: str = "yaml",
        prompt_config: Optional[Dict[str, Any]] = None
    ):
        self._prompt_manager = prompt_manager
        self._prompt_backend = prompt_backend
        self._prompt_config = prompt_config or {}

    @property
    def prompt_manager(self) -> PromptManager:
        """Get prompt manager (lazy initialization)."""
        if self._prompt_manager is None:
            from graflow.prompts.factory import PromptManagerFactory
            self._prompt_manager = PromptManagerFactory.create(
                self._prompt_backend,
                **self._prompt_config
            )
        return self._prompt_manager
```

### Usage in Tasks

```python
from graflow.core.decorators import task
from graflow.core.context import ExecutionContext

@task(inject_context=True)
def send_welcome_email(context: ExecutionContext, user_name: str, user_email: str):
    # Get prompt from context
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

### Workflow Configuration

```python
from graflow.core.workflow import workflow

with workflow("onboarding_workflow") as wf:
    # Configure prompt manager
    wf.context.prompt_backend = "yaml"
    wf.context.prompt_config = {"prompts_dir": "./prompts"}

    # Or set directly
    from graflow.prompts import PromptManagerFactory
    wf.context.prompt_manager = PromptManagerFactory.create(
        "yaml",
        prompts_dir="./prompts"
    )

    # Use in tasks
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

### Example 2: LangFuse Backend

```python
from graflow.prompts import PromptManagerFactory

# Create LangFuse manager (uses env vars for credentials)
pm = PromptManagerFactory.create("langfuse")

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

with workflow("customer_engagement") as wf:
    # Configure prompt backend
    wf.context.prompt_backend = "yaml"
    wf.context.prompt_config = {"prompts_dir": "./prompts"}

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

## Custom Exceptions

```python
# graflow/prompts/exceptions.py

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
    """Raised when prompt manager configuration is invalid."""
    pass
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
    "langfuse>=3.8.1",  # Already present - LangFuse backend
]
```

### Installation

```bash
# Basic (YAML backend only)
uv pip install graflow

# With LangFuse support
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

### LangFuse Backend

```bash
# Required for LangFuse backend
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...

# Optional: Custom LangFuse host
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
   - `_load_all_prompts()` - Recursively scan directories
   - `_load_prompt_file()` - Parse YAML to Prompt objects
   - `get_prompt()` - Fetch with caching
   - `list_prompts()`, `list_labels()` - Listing methods
2. Add cache implementation with TTL support
3. Test with various directory structures

### Phase 3: LangFuse Backend
1. Implement `langfuse_manager.py`
   - Optional import with `LANGFUSE_AVAILABLE` flag
   - `get_prompt()` - Fetch from LangFuse with caching
   - Credential loading from env vars
2. Convert LangFuse prompt objects to PromptVersion

### Phase 4: Factory & Public API
1. Implement `factory.py` (PromptManagerFactory)
   - Backend registration
   - `create()` method
   - Conditional LangFuse registration
2. Implement `__init__.py` (public API exports)

### Phase 5: ExecutionContext Integration
1. Modify `graflow/core/context.py`
   - Add `prompt_manager`, `prompt_backend`, `prompt_config` parameters
   - Add lazy `prompt_manager` property
   - Update `create()` classmethod

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
2. ✅ LangFuse backend fetches prompts from LangFuse server
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
2. **LangFuse compatibility**: Same naming conventions and API patterns
3. **Flexible organization**: Virtual folders map to filesystem directories
4. **Version control**: Labels for environments, numeric versions for tracking
5. **Smart caching**: Per-call TTL control with sensible defaults
6. **Easy configuration**: Optional prompts_dir with env var fallback
7. **Seamless integration**: Lazy-loaded via ExecutionContext
8. **Production-ready**: Error handling, caching, and type safety

The module enables teams to:
- Develop locally with YAML files
- Deploy to production with LangFuse
- Version prompts for A/B testing and experiments
- Integrate prompts seamlessly into Graflow workflows
