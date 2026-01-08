"""Tests for LLM dependency injection in @task decorator."""

import pytest

from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph
from graflow.core.workflow import workflow
from graflow.llm.agents.base import LLMAgent
from graflow.llm.client import LLMClient

# Skip tests if litellm is not available
pytest.importorskip("litellm")


class MockLLMAgent(LLMAgent):
    """Mock LLM agent for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name

    def run(self, input_text: str, **kwargs):
        return {"output": f"Mock response to: {input_text}", "steps": [], "metadata": {}}

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
        def test_task(llm_client: LLMClient) -> str:
            return f"Model: {llm_client.model}"

        # Create LLMClient and inject
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(graph=graph, llm_client=llm_client)

        # Set execution context
        test_task.set_execution_context(context)

        # Call task
        result = test_task()
        assert result == "Model: gpt-4o-mini"

    def test_inject_llm_client_shared_instance(self):
        """Test that LLMClient instance is shared across tasks."""
        from graflow.llm import LLMClient

        @task(inject_llm_client=True)
        def task_a(llm_client: LLMClient) -> int:
            return id(llm_client)

        @task(inject_llm_client=True)
        def task_b(llm_client: LLMClient) -> int:
            return id(llm_client)

        # Create context with LLMClient
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(graph=graph, llm_client=llm_client)

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
        def test_task(llm_client: LLMClient) -> tuple[str, float | None]:
            return llm_client.model, llm_client.default_params.get("temperature")

        # Create context with direct client injection
        graph = TaskGraph()
        context = ExecutionContext.create(
            graph=graph,
            llm_client=custom_client,  # Direct injection
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
        def test_task(llm_client: LLMClient) -> str:
            # Verify LLMClient is auto-created
            assert llm_client is not None
            assert isinstance(llm_client, LLMClient)
            # Default model should be gpt-5-mini (or GRAFLOW_LLM_MODEL env var)
            assert llm_client.model in ["gpt-5-mini", "gpt-4o", "claude-3-5-sonnet-20241022"]
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
        def test_task(llm_agent: LLMAgent) -> dict:
            return llm_agent.run("Hello")

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
        def test_task(llm_agent: LLMAgent) -> str:
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


class TestWorkflowContextAgentRegistration:
    """Ensure WorkflowContext-level agent registration works."""

    def test_workflow_registers_factory_agent(self):
        """Factories should run after ExecutionContext is created."""
        with workflow("agent_factory_workflow") as wf:
            factory_calls: list[str] = []

            def factory(exec_context: ExecutionContext) -> LLMAgent:
                factory_calls.append(exec_context.session_id)
                return MockLLMAgent(name="factory_agent")

            wf.register_llm_agent("factory_agent", factory)

            @task(inject_llm_agent="factory_agent")
            def run_research(llm_agent: LLMAgent) -> str:
                return llm_agent.run("ping")["output"]

            result = wf.execute("run_research")

        assert factory_calls, "Factory should be invoked once during execution"
        assert "Mock response to: ping" in result

    def test_workflow_registers_prebuilt_agent(self):
        """Pre-built agents can be registered directly on the workflow."""
        agent = MockLLMAgent(name="prebuilt")

        with workflow("agent_instance_workflow") as wf:
            wf.register_llm_agent("prebuilt_agent", agent)

            @task(inject_llm_agent="prebuilt_agent")
            def use_agent(llm_agent: LLMAgent) -> str:
                return llm_agent.name

            result = wf.execute("use_agent")

        assert result == "prebuilt"


class TestTaskExecutionContextLLMAccessors:
    """Test LLM accessors through TaskExecutionContext (inject_context=True)."""

    def test_context_llm_client_accessor(self):
        """Test accessing LLMClient through TaskExecutionContext.llm_client."""
        from graflow.llm import LLMClient

        @task(inject_context=True)
        def test_task(context: TaskExecutionContext) -> str:
            # Access LLMClient through context
            llm = context.llm_client
            assert llm is not None
            assert isinstance(llm, LLMClient)
            return llm.model

        # Create context with LLMClient
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(graph=graph, llm_client=llm_client)

        # Set execution context
        test_task.set_execution_context(context)

        # Should access LLMClient through context
        result = test_task()
        assert result == "gpt-4o-mini"

    def test_context_llm_client_accessor_shared_instance(self):
        """Test that context.llm_client returns the same instance."""
        from graflow.llm import LLMClient

        @task(inject_context=True)
        def task_a(context: TaskExecutionContext) -> int:
            return id(context.llm_client)

        @task(inject_context=True)
        def task_b(context: TaskExecutionContext) -> int:
            return id(context.llm_client)

        # Create context with LLMClient
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(graph=graph, llm_client=llm_client)

        # Set execution context
        task_a.set_execution_context(context)
        task_b.set_execution_context(context)

        # Both should return same instance
        id_a = task_a()
        id_b = task_b()
        assert id_a == id_b

    def test_context_get_llm_agent_accessor(self):
        """Test accessing LLMAgent through TaskExecutionContext.get_llm_agent()."""

        @task(inject_context=True)
        def test_task(context: TaskExecutionContext) -> dict:
            # Access agent through context
            agent = context.get_llm_agent("test_agent")
            return agent.run("Hello")

        # Create context and register agent
        graph = TaskGraph()
        context = ExecutionContext.create(graph=graph)
        agent = MockLLMAgent(name="test_agent")
        context.register_llm_agent("test_agent", agent)

        # Set execution context
        test_task.set_execution_context(context)

        # Should access agent through context
        result = test_task()
        assert result["output"] == "Mock response to: Hello"

    def test_context_get_llm_agent_not_registered_raises_error(self):
        """Test context.get_llm_agent() raises KeyError for unregistered agent."""

        @task(inject_context=True)
        def test_task(context: TaskExecutionContext) -> None:
            # Try to access non-existent agent
            context.get_llm_agent("missing_agent")

        # Create context without registering agent
        graph = TaskGraph()
        context = ExecutionContext.create(graph=graph)

        # Set execution context
        test_task.set_execution_context(context)

        # Should raise KeyError
        with pytest.raises(KeyError, match="not found in registry"):
            test_task()

    def test_context_mixed_llm_access(self):
        """Test accessing both LLMClient and LLMAgent through context."""
        from graflow.llm import LLMClient

        @task(inject_context=True)
        def test_task(context: TaskExecutionContext) -> dict[str, str]:
            # Access both LLMClient and LLMAgent
            llm = context.llm_client
            agent = context.get_llm_agent("test_agent")

            return {"model": llm.model, "agent_response": agent.run("test")["output"]}

        # Create context with both LLMClient and agent
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(graph=graph, llm_client=llm_client)
        agent = MockLLMAgent(name="test_agent")
        context.register_llm_agent("test_agent", agent)

        # Set execution context
        test_task.set_execution_context(context)

        # Should access both through context
        result = test_task()
        assert result["model"] == "gpt-4o-mini"
        assert result["agent_response"] == "Mock response to: test"


class TestMultipleInjectionTypes:
    """Test that multiple injection types can co-exist."""

    def test_inject_client_and_agent_together(self):
        """Test injecting both LLMClient and LLMAgent into the same task."""
        from graflow.llm import LLMClient

        @task(inject_llm_client=True, inject_llm_agent="test_agent")
        def test_task(llm_client: LLMClient, llm_agent: LLMAgent) -> dict[str, str]:
            # Both should be injected as named arguments
            return {"model": llm_client.model, "agent_name": llm_agent.name}

        # Create context with both client and agent
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(graph=graph, llm_client=llm_client)
        agent = MockLLMAgent(name="test_agent")
        context.register_llm_agent("test_agent", agent)

        # Set execution context
        test_task.set_execution_context(context)

        # Should inject both
        result = test_task()
        assert result["model"] == "gpt-4o-mini"
        assert result["agent_name"] == "test_agent"

    def test_inject_context_and_client_together(self):
        """Test injecting both TaskExecutionContext and LLMClient."""
        from graflow.llm import LLMClient

        @task(inject_context=True, inject_llm_client=True)
        def test_task(context: TaskExecutionContext, llm_client: LLMClient) -> dict[str, str | bool]:
            # Context is positional, llm_client is named argument
            return {"has_context": context is not None, "model": llm_client.model}

        # Create context with client
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(graph=graph, llm_client=llm_client)

        # Set execution context
        test_task.set_execution_context(context)

        # Should inject both
        result = test_task()
        assert result["has_context"] is True
        assert result["model"] == "gpt-4o-mini"

    def test_inject_all_three_together(self):
        """Test injecting TaskExecutionContext, LLMClient, and LLMAgent together."""
        from graflow.llm import LLMClient

        @task(inject_context=True, inject_llm_client=True, inject_llm_agent="test_agent")
        def test_task(
            context: TaskExecutionContext, llm_client: LLMClient, llm_agent: LLMAgent
        ) -> dict[str, str | bool]:
            # All three should be injected
            return {
                "has_context": context is not None,
                "model": llm_client.model,
                "agent_response": llm_agent.run("Hello")["output"],
            }

        # Create context with client and agent
        graph = TaskGraph()
        llm_client = LLMClient(model="gpt-4o-mini")
        context = ExecutionContext.create(graph=graph, llm_client=llm_client)
        agent = MockLLMAgent(name="test_agent")
        context.register_llm_agent("test_agent", agent)

        # Set execution context
        test_task.set_execution_context(context)

        # Should inject all three
        result = test_task()
        assert result["has_context"] is True
        assert result["model"] == "gpt-4o-mini"
        assert result["agent_response"] == "Mock response to: Hello"
