# Prompt Management Examples

This directory demonstrates Graflow's prompt management module for managing LLM prompts.

## Examples

### yaml_prompts.py
Basic usage of `YAMLPromptManager` for loading and rendering prompts from YAML files.

```bash
make py examples/14_prompt_management/yaml_prompts.py
```

### langfuse_prompts.py
Using `LangfusePromptManager` for cloud-based prompt management.

**Prerequisites:**
1. Install langfuse: `pip install langfuse` (or `pip install graflow[tracing]`)
2. Create prompts in Langfuse dashboard:
   - `greeting` (text): `"Hello {{name}}, welcome to {{product}}!"`
   - `assistant` (chat): system + user messages with `{{task}}` variable

```bash
# Set credentials
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=http://localhost:3000  # For local Langfuse

make py examples/14_prompt_management/langfuse_prompts.py
```

### workflow_with_prompts.py
Using prompts within Graflow tasks via context injection.

```bash
make py examples/14_prompt_management/workflow_with_prompts.py
```

Key pattern:
```python
from graflow.core.decorators import task
from graflow.core.context import TaskExecutionContext
from graflow.core.workflow import workflow
from graflow.prompts.factory import PromptManagerFactory

# Create workflow with prompt manager
pm = PromptManagerFactory.create("yaml", prompts_dir="./prompts")

with workflow("my_workflow", prompt_manager=pm) as ctx:

    @task(inject_context=True)
    def setup(context: TaskExecutionContext):
        channel = context.get_channel()
        channel.set("name", "Alice")

    @task(inject_context=True)
    def greet(context: TaskExecutionContext):
        pm = context.prompt_manager
        channel = context.get_channel()
        name = channel.get("name")
        prompt = pm.get_text_prompt("greeting")
        return prompt.render(name=name)

    setup >> greet
    ctx.execute("setup")
```

## YAML Prompt Format

Prompts are stored in YAML files with the following structure:

```yaml
prompt_name:
  type: text  # or "chat"
  labels:
    production:
      content: "Hello {{name}}!"
      version: 1
      created_at: "2024-01-01T10:00:00"
      metadata:
        author: "team@example.com"
    staging:
      content: "Hi {{name}}!"
      version: 2
```

### Text Prompts
- Single string content with `{{variable}}` placeholders
- Use `render(var1=val1, var2=val2)` to substitute variables

### Chat Prompts
- List of messages with `role` and `content` fields
- Suitable for LLM conversation APIs

```yaml
assistant:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are a helpful assistant."
        - role: user
          content: "Help me with {{task}}."
```

## Key Features

- **Label-based access**: `get_prompt("name", label="production")`
- **Version-based access**: `get_prompt("name", version=1)`
- **Auto-reload**: Modified YAML files are reloaded automatically
- **Subdirectory support**: Organize prompts in folders (e.g., `customer/welcome`)
