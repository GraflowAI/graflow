"""Tests for LLM dependency injection in @task decorator."""

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph
from graflow.llm.agents.base import LLMAgent
from graflow.llm.client import LLMClient

# Skip tests if litellm is not available
pytest.importorskip("litellm")


class MockLLMAgent(LLMAgent):
    """Mock LLM agent for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name

    def run(self, input_text: str, **kwargs):
        return {
            "output": f"Mock response to: {input_text}",
            "steps": [],
            "metadata": {}
        }

    def stream(self, input_text: str, **kwargs):
        yield f"Mock response to: {input_text}"

    @property
    def name(self) -> str:
        return self._name


class TestLLMClientInjection:
    """Test LLMClient injection in tasks."""

    def test_inject_llm_client_basic(self):
        """Test basic LLMClient injection."""
        from graflow.llm import LLMClient

        @task(inject_llm_client=True)
        def test_task(llm):
            return f"Model: {llm.model}"

        # Create LLMClient and inject
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(
            graph=graph,
            llm_client=llm_client
        )

        # Set execution context
        test_task.set_execution_context(context)

        # Call task
        result = test_task()
        assert result == "Model: gpt-4o-mini"

    def test_inject_llm_client_shared_instance(self):
        """Test that LLMClient instance is shared across tasks."""
        from graflow.llm import LLMClient

        @task(inject_llm_client=True)
        def task_a(llm):
            return id(llm)

        @task(inject_llm_client=True)
        def task_b(llm):
            return id(llm)

        # Create context with LLMClient
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(
            graph=graph,
            llm_client=llm_client
        )

        # Set execution context
        task_a.set_execution_context(context)
        task_b.set_execution_context(context)

        # Both tasks should get the same LLMClient instance
        id_a = task_a()
        id_b = task_b()
        assert id_a == id_b

    def test_inject_llm_client_direct_injection(self):
        """Test direct LLMClient injection."""
        from graflow.llm import LLMClient

        # Create custom client
        custom_client = LLMClient(model="custom-model", temperature=0.9)

        @task(inject_llm_client=True)
        def test_task(llm):
            return llm.model, llm.default_params.get("temperature")

        # Create context with direct client injection
        graph = TaskGraph()
        context = ExecutionContext.create(
            graph=graph,
            llm_client=custom_client  # Direct injection
        )

        # Set execution context
        test_task.set_execution_context(context)

        # Should use injected client
        model, temp = test_task()
        assert model == "custom-model"
        assert temp == 0.9

    def test_inject_llm_client_auto_creates_default(self):
        """Test that LLMClient is auto-created with defaults if not set."""
        @task(inject_llm_client=True)
        def test_task(llm):
            # Verify LLMClient is auto-created
            assert llm is not None
            assert isinstance(llm, LLMClient)
            # Default model should be gpt-5-mini (or GRAFLOW_LLM_MODEL env var)
            assert llm.model in ["gpt-5-mini", "gpt-4o", "claude-3-5-sonnet-20241022"]
            return "success"

        # Create context without explicit LLMClient
        graph = TaskGraph()
        context = ExecutionContext.create(graph=graph)

        # Set execution context
        test_task.set_execution_context(context)

        # Should auto-create default LLMClient and succeed
        result = test_task()
        assert result == "success"


class TestLLMAgentInjection:
    """Test LLMAgent injection in tasks."""

    def test_inject_llm_agent_basic(self):
        """Test basic LLMAgent injection."""
        @task(inject_llm_agent="test_agent")
        def test_task(agent):
            return agent.run("Hello")

        # Create context and register agent
        graph = TaskGraph()
        context = ExecutionContext.create(graph=graph)
        agent = MockLLMAgent(name="test_agent")
        context.register_llm_agent("test_agent", agent)

        # Set execution context
        test_task.set_execution_context(context)

        # Call task
        result = test_task()
        assert result["output"] == "Mock response to: Hello"

    def test_inject_llm_agent_not_registered_raises_error(self):
        """Test that injection fails if agent not registered."""
        @task(inject_llm_agent="missing_agent")
        def test_task(agent):
            return "Should not reach here"

        # Create context without registering agent
        graph = TaskGraph()
        context = ExecutionContext.create(graph=graph)

        # Set execution context
        test_task.set_execution_context(context)

        # Should raise error
        with pytest.raises(RuntimeError, match="not registered"):
            test_task()


class TestAgentRegistry:
    """Test agent registry in ExecutionContext."""

    def test_register_and_get_agent(self):
        """Test registering and retrieving agent."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph=graph)

        # Register agent
        agent = MockLLMAgent(name="test")
        context.register_llm_agent("test", agent)

        # Retrieve agent
        retrieved = context.get_llm_agent("test")
        assert retrieved is agent
        assert retrieved.name == "test"

    def test_get_nonexistent_agent_raises_error(self):
        """Test getting non-existent agent raises KeyError."""
        graph = TaskGraph()
        context = ExecutionContext.create(graph=graph)

        with pytest.raises(KeyError, match="not found in registry"):
            context.get_llm_agent("nonexistent")
