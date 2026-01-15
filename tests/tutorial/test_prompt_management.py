"""
Unit tests for Prompt Management from Tasks and Workflows Guide.

Tests prompt manager injection and usage features documented in the guide.
Uses temporary YAML files to avoid external dependencies.
"""

from pathlib import Path

import pytest

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.prompts.exceptions import PromptCompilationError
from graflow.prompts.factory import PromptManagerFactory
from graflow.prompts.models import ChatPrompt, TextPrompt


class TestPromptManagerInjection:
    """Tests for prompt_manager injection via workflow context"""

    @pytest.fixture
    def prompts_dir(self, tmp_path: Path) -> Path:
        """Create temporary prompts directory with sample YAML files."""
        # Create greetings.yaml
        greetings_yaml = tmp_path / "greetings.yaml"
        greetings_yaml.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}, welcome to {{product}}!"
      version: 1
    staging:
      content: "Hi {{name}}! Testing {{product}}."
      version: 2

farewell:
  type: text
  labels:
    production:
      content: "Goodbye {{name}}, thanks for using {{product}}!"
      version: 1
""")

        # Create agents.yaml
        agents_yaml = tmp_path / "agents.yaml"
        agents_yaml.write_text("""
assistant:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are a helpful assistant specializing in {{domain}}."
        - role: user
          content: "Please help me with {{task}}."
      version: 1
""")
        return tmp_path

    def test_prompt_manager_basic_injection(self, prompts_dir: Path):
        """Test basic prompt manager injection via workflow context"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        with workflow("prompt_test", prompt_manager=pm) as ctx:

            @task(inject_context=True)
            def use_prompt(context: TaskExecutionContext):
                pm = context.prompt_manager
                prompt = pm.get_text_prompt("greeting", label="production")
                return prompt.render(name="Alice", product="Graflow")

            _, exec_context = ctx.execute(ret_context=True)

        result = exec_context.get_result("use_prompt")
        assert result == "Hello Alice, welcome to Graflow!"

    def test_prompt_manager_text_prompt_labels(self, prompts_dir: Path):
        """Test accessing different prompt labels"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        with workflow("label_test", prompt_manager=pm) as ctx:

            @task(inject_context=True)
            def get_production(context: TaskExecutionContext):
                pm = context.prompt_manager
                prompt = pm.get_text_prompt("greeting", label="production")
                return prompt.render(name="Bob", product="Test")

            @task(inject_context=True)
            def get_staging(context: TaskExecutionContext):
                pm = context.prompt_manager
                prompt = pm.get_text_prompt("greeting", label="staging")
                return prompt.render(name="Bob", product="Test")

            get_production >> get_staging  # type: ignore

            _, exec_context = ctx.execute(ret_context=True)

        assert exec_context.get_result("get_production") == "Hello Bob, welcome to Test!"
        assert exec_context.get_result("get_staging") == "Hi Bob! Testing Test."

    def test_prompt_manager_chat_prompt(self, prompts_dir: Path):
        """Test chat prompt rendering"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        with workflow("chat_test", prompt_manager=pm) as ctx:

            @task(inject_context=True)
            def get_chat_messages(context: TaskExecutionContext):
                pm = context.prompt_manager
                prompt = pm.get_chat_prompt("assistant", label="production")
                messages = prompt.render(domain="Python", task="debugging")
                return messages

            _, exec_context = ctx.execute(ret_context=True)

        messages = exec_context.get_result("get_chat_messages")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Python" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "debugging" in messages[1]["content"]

    def test_prompt_manager_multiple_tasks(self, prompts_dir: Path):
        """Test prompt manager access across multiple tasks"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        with workflow("multi_task_test", prompt_manager=pm) as ctx:

            @task(inject_context=True)
            def task1(context: TaskExecutionContext):
                pm = context.prompt_manager
                prompt = pm.get_text_prompt("greeting", label="production")
                return prompt.render(name="User1", product="App")

            @task(inject_context=True)
            def task2(context: TaskExecutionContext):
                pm = context.prompt_manager
                prompt = pm.get_text_prompt("farewell", label="production")
                return prompt.render(name="User1", product="App")

            task1 >> task2  # type: ignore

            _, exec_context = ctx.execute(ret_context=True)

        assert exec_context.get_result("task1") == "Hello User1, welcome to App!"
        assert exec_context.get_result("task2") == "Goodbye User1, thanks for using App!"


class TestPromptManagerWithChannels:
    """Tests for prompt manager combined with channel data sharing"""

    @pytest.fixture
    def prompts_dir(self, tmp_path: Path) -> Path:
        """Create temporary prompts directory."""
        greetings_yaml = tmp_path / "greetings.yaml"
        greetings_yaml.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}, welcome to {{product}}!"
      version: 1

farewell:
  type: text
  labels:
    production:
      content: "Goodbye {{name}}, thanks for using {{product}}!"
      version: 1
""")

        agents_yaml = tmp_path / "agents.yaml"
        agents_yaml.write_text("""
assistant:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are a helpful assistant specializing in {{domain}}."
        - role: user
          content: "Please help me with {{task}}."
      version: 1
""")
        return tmp_path

    def test_workflow_with_channel_parameters(self, prompts_dir: Path):
        """Test workflow pattern with setup task storing parameters in channel"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        with workflow("channel_params", prompt_manager=pm) as ctx:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext):
                channel = context.get_channel()
                channel.set("customer_name", "Alice")
                channel.set("product_name", "Graflow")

            @task(inject_context=True)
            def greet_customer(context: TaskExecutionContext):
                pm = context.prompt_manager
                channel = context.get_channel()

                name = channel.get("customer_name")
                product = channel.get("product_name")

                prompt = pm.get_text_prompt("greeting", label="production")
                greeting = prompt.render(name=name, product=product)
                channel.set("greeting", greeting)
                return greeting

            setup >> greet_customer  # type: ignore

            _, exec_context = ctx.execute(ret_context=True)

        channel = exec_context.channel
        assert channel.get("greeting") == "Hello Alice, welcome to Graflow!"

    def test_full_onboarding_workflow(self, prompts_dir: Path):
        """Test complete onboarding workflow pattern from example"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        with workflow("customer_onboarding", prompt_manager=pm) as ctx:

            @task(inject_context=True)
            def setup(context: TaskExecutionContext):
                channel = context.get_channel()
                channel.set("customer_name", "Alice")
                channel.set("product_name", "Graflow")
                channel.set("domain", "Python")
                channel.set("task_description", "onboarding")

            @task(inject_context=True)
            def greet_customer(context: TaskExecutionContext) -> str:
                pm = context.prompt_manager
                channel = context.get_channel()

                customer_name = channel.get("customer_name")
                product_name = channel.get("product_name")

                prompt = pm.get_text_prompt("greeting", label="production")
                greeting = prompt.render(name=customer_name, product=product_name)
                channel.set("greeting", greeting)
                return greeting

            @task(inject_context=True)
            def generate_assistant_messages(context: TaskExecutionContext) -> list:
                pm = context.prompt_manager
                channel = context.get_channel()

                domain = channel.get("domain")
                task_description = channel.get("task_description")

                prompt = pm.get_chat_prompt("assistant", label="production")
                messages = prompt.render(domain=domain, task=task_description)
                channel.set("messages", messages)
                return messages

            @task(inject_context=True)
            def send_farewell(context: TaskExecutionContext) -> str:
                pm = context.prompt_manager
                channel = context.get_channel()

                customer_name = channel.get("customer_name")
                product_name = channel.get("product_name")

                prompt = pm.get_text_prompt("farewell", label="production")
                farewell = prompt.render(name=customer_name, product=product_name)
                channel.set("farewell", farewell)
                return farewell

            setup >> greet_customer >> generate_assistant_messages >> send_farewell  # type: ignore

            _, exec_context = ctx.execute("setup", ret_context=True)

        # Verify results from channel
        channel = exec_context.channel
        assert channel.get("greeting") == "Hello Alice, welcome to Graflow!"
        assert channel.get("farewell") == "Goodbye Alice, thanks for using Graflow!"

        messages = channel.get("messages")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Python" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "onboarding" in messages[1]["content"]


class TestPromptModels:
    """Tests for prompt model classes"""

    def test_text_prompt_render(self):
        """Test TextPrompt rendering"""
        prompt = TextPrompt(
            name="test",
            content="Hello {{name}}!",
            version=1,
        )
        result = prompt.render(name="World")
        assert result == "Hello World!"

    def test_text_prompt_render_multiple_vars(self):
        """Test TextPrompt with multiple variables"""
        prompt = TextPrompt(
            name="test",
            content="{{greeting}} {{name}}, welcome to {{place}}!",
            version=1,
        )
        result = prompt.render(greeting="Hi", name="Alice", place="Graflow")
        assert result == "Hi Alice, welcome to Graflow!"

    def test_chat_prompt_render(self):
        """Test ChatPrompt rendering"""
        prompt = ChatPrompt(
            name="test",
            content=[
                {"role": "system", "content": "You are an expert in {{domain}}."},
                {"role": "user", "content": "Help me with {{task}}."},
            ],
            version=1,
        )
        messages = prompt.render(domain="Python", task="debugging")
        assert len(messages) == 2
        assert messages[0]["content"] == "You are an expert in Python."
        assert messages[1]["content"] == "Help me with debugging."

    def test_text_prompt_missing_variable_raises(self):
        """Test that missing variable raises PromptCompilationError"""
        prompt = TextPrompt(
            name="test",
            content="Hello {{name}}!",
            version=1,
        )
        with pytest.raises(PromptCompilationError):
            prompt.render()  # Missing 'name'


class TestYAMLPromptManagerStandalone:
    """Tests for YAML prompt manager standalone usage (yaml_prompts.py patterns)"""

    @pytest.fixture
    def prompts_dir(self, tmp_path: Path) -> Path:
        """Create temporary prompts directory with sample YAML files."""
        greetings_yaml = tmp_path / "greetings.yaml"
        greetings_yaml.write_text("""
greeting:
  type: text
  labels:
    production:
      content: "Hello {{name}}, welcome to {{product}}!"
      version: 2
      created_at: "2024-01-15T10:00:00"
      metadata:
        author: "team@example.com"
    staging:
      content: "Hi {{name}}! Testing {{product}} features."
      version: 1
      created_at: "2024-01-10T10:00:00"

farewell:
  type: text
  labels:
    production:
      content: "Goodbye {{name}}, thanks for using {{product}}!"
      version: 1
""")

        agents_yaml = tmp_path / "agents.yaml"
        agents_yaml.write_text("""
assistant:
  type: chat
  labels:
    production:
      content:
        - role: system
          content: "You are a helpful assistant specializing in {{domain}}."
        - role: user
          content: "Please help me with {{task}}."
      version: 1
      metadata:
        model: "gpt-4o-mini"
""")
        return tmp_path

    def test_get_text_prompt_default_label(self, prompts_dir: Path):
        """Test getting text prompt with default label (production)"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        # Default label should be 'production'
        greeting = pm.get_text_prompt("greeting")
        rendered = greeting.render(name="Alice", product="Graflow")
        assert rendered == "Hello Alice, welcome to Graflow!"

    def test_get_text_prompt_by_label(self, prompts_dir: Path):
        """Test getting text prompt by specific label"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        # Get staging label
        staging_greeting = pm.get_text_prompt("greeting", label="staging")
        rendered = staging_greeting.render(name="Alice", product="Graflow")
        assert rendered == "Hi Alice! Testing Graflow features."

    def test_get_chat_prompt_default_label(self, prompts_dir: Path):
        """Test getting chat prompt with default label"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        assistant = pm.get_chat_prompt("assistant")
        messages = assistant.render(domain="Python", task="debugging")

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "Python" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "debugging" in messages[1]["content"]

    def test_prompt_metadata_access(self, prompts_dir: Path):
        """Test accessing prompt metadata"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        prompt = pm.get_prompt("greeting")
        assert prompt.name == "greeting"
        assert prompt.label == "production"
        assert prompt.version == 2
        assert prompt.created_at == "2024-01-15T10:00:00"
        assert prompt.metadata == {"author": "team@example.com"}

    def test_prompt_version_info(self, prompts_dir: Path):
        """Test prompt version information"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        # Production version should be 2
        prod = pm.get_text_prompt("greeting", label="production")
        assert prod.version == 2

        # Staging version should be 1
        staging = pm.get_text_prompt("greeting", label="staging")
        assert staging.version == 1

    def test_multiple_prompts_same_file(self, prompts_dir: Path):
        """Test loading multiple prompts from same YAML file"""
        pm = PromptManagerFactory.create("yaml", prompts_dir=str(prompts_dir))

        # Both greeting and farewell are in greetings.yaml
        greeting = pm.get_text_prompt("greeting")
        farewell = pm.get_text_prompt("farewell")

        assert greeting.render(name="Bob", product="App") == "Hello Bob, welcome to App!"
        assert farewell.render(name="Bob", product="App") == "Goodbye Bob, thanks for using App!"


class TestPromptManagerFactory:
    """Tests for PromptManagerFactory"""

    def test_create_yaml_manager(self, tmp_path: Path):
        """Test creating YAML prompt manager via factory"""
        # Create a simple prompt file
        prompt_file = tmp_path / "test.yaml"
        prompt_file.write_text("""
test:
  type: text
  labels:
    production:
      content: "Test content"
      version: 1
""")

        pm = PromptManagerFactory.create("yaml", prompts_dir=str(tmp_path))
        prompt = pm.get_text_prompt("test", label="production")
        assert prompt.content == "Test content"

    def test_factory_unknown_backend_raises(self):
        """Test that unknown backend raises ValueError"""
        with pytest.raises(ValueError, match="Unknown backend"):
            PromptManagerFactory.create("unknown_backend")
