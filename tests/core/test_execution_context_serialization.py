"""Comprehensive tests for ExecutionContext serialization and deserialization.

This module contains extensive test coverage for the serialization/deserialization
capabilities of ExecutionContext objects, which is essential for:

- Distributed task execution across multiple processes
- Persistent storage of execution state for resumption
- Inter-process communication in worker-based architectures
- Fault tolerance and state recovery mechanisms

The tests validate that all ExecutionContext components (channels, queues, managers,
state) can survive pickle operations while maintaining data integrity and functionality.
"""

import os
import pickle
import tempfile

import pytest

from graflow.channels.base import Channel
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph
from graflow.core.task import Task


# Global functions for testing function manager serialization
@task
def global_test_func():
    return "test"


@task
def global_math_func(x, y):
    return x + y


# Global functions for testing queue backend serialization
def global_task_func_1():
    return "task1_result"


def global_task_func_2():
    return "task2_result"


class TestExecutionContextSerialization:
    """Comprehensive test suite for ExecutionContext serialization and deserialization.

    This test class validates the complete serialization/deserialization pipeline for
    ExecutionContext objects, ensuring that all components can survive pickle operations
    and maintain their functionality after reconstruction. Key areas covered include:

    - Core ExecutionContext state preservation
    - Channel backend compatibility (Memory and Redis)
    - Task queue and execution state management
    - Function registry and manager persistence
    - Error handling and edge case robustness
    - Cross-platform serialization compatibility

    These tests are critical for distributed execution scenarios where ExecutionContext
    objects need to be transferred between processes or persisted to disk.
    """

    def test_basic_serialization(self):
        """Test basic ExecutionContext serialization and deserialization.

        This test verifies that:
        - ExecutionContext can be pickled and unpickled successfully
        - Core attributes (start_node, max_steps, session_id) are preserved
        - Execution state (steps, results) survives the serialization process
        - Basic functionality remains intact after reconstruction
        """
        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        context = ExecutionContext.create(
            graph=graph, start_node="test_task", max_steps=15, default_max_retries=5, channel_backend="memory"
        )

        # Set some state
        context.set_result("test_task", "test_result")
        context.increment_step()
        context.increment_step()

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save context
            context.save(context_path)

            # Load context
            loaded_context = ExecutionContext.load(context_path)

            # Verify basic attributes
            assert loaded_context.start_node == "test_task"
            assert loaded_context.max_steps == 15
            assert loaded_context.default_max_retries == 5
            assert loaded_context.steps == 2
            assert loaded_context.session_id == context.session_id

            # Verify result is preserved
            assert loaded_context.get_result("test_task") == "test_result"

    def test_memory_channel_reconstruction(self):
        """Test that memory channel is properly reconstructed after deserialization.

        This test ensures that:
        - MemoryChannel instances can survive pickle serialization
        - Channel data (including complex nested objects) is preserved
        - The reconstructed channel maintains full functionality
        - No data corruption occurs during the serialization process
        - Channel type and interface remain consistent after deserialization
        """
        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        context = ExecutionContext.create(graph=graph, start_node="test_task", max_steps=10, channel_backend="memory")

        # Set some channel data
        context.channel.set("test_key", "test_value")
        context.channel.set("complex_data", {"nested": {"data": [1, 2, 3]}})

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify channel type and functionality
            assert loaded_context.channel is not None
            assert isinstance(loaded_context.channel, Channel)

            # Verify channel data is preserved
            assert loaded_context.channel.get("test_key") == "test_value"
            assert loaded_context.channel.get("complex_data") == {"nested": {"data": [1, 2, 3]}}

            # Verify channel is functional after reconstruction
            loaded_context.channel.set("new_key", "new_value")
            assert loaded_context.channel.get("new_key") == "new_value"

    def test_redis_channel_reconstruction(self, clean_redis):
        """Test that Redis channel is properly reconstructed after deserialization.

        This test validates that:
        - RedisChannel survives pickle serialization despite connection pool complexity
        - Redis connection parameters are preserved and restored correctly
        - Channel data persists in Redis and remains accessible after reconstruction
        - New Redis connections are established automatically during deserialization
        - Channel functionality (set/get operations) works seamlessly after reload
        - Uses Docker-based Redis server for isolated testing environment
        """
        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        # Use the clean Redis server from Docker fixture
        # Extract connection parameters for serialization compatibility
        redis_config = {
            "host": clean_redis.connection_pool.connection_kwargs.get("host", "localhost"),
            "port": clean_redis.connection_pool.connection_kwargs.get("port", 6379),
            "db": clean_redis.connection_pool.connection_kwargs.get("db", 0),
        }

        # Create context with Redis backend using the Docker Redis
        context = ExecutionContext.create(
            graph=graph, start_node="test_task", max_steps=10, channel_backend="redis", config=redis_config
        )

        # Set some channel data
        test_key = f"test_ser_{context.session_id}"
        context.channel.set(test_key, "redis_test_value")

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify channel type and functionality
            assert loaded_context.channel is not None
            assert isinstance(loaded_context.channel, Channel)

            # Verify channel data is preserved (should exist in Redis)
            assert loaded_context.channel.get(test_key) == "redis_test_value"

            # Verify channel is functional after reconstruction
            new_key = f"test_new_{loaded_context.session_id}"
            loaded_context.channel.set(new_key, "new_redis_value")
            assert loaded_context.channel.get(new_key) == "new_redis_value"

    def test_channel_backend_config_preservation(self):
        """Test that channel backend and config are preserved during serialization.

        This test confirms that:
        - Custom channel configuration parameters survive serialization
        - Channel backend type selection is maintained after deserialization
        - Configuration-specific behavior remains consistent
        - Custom settings (timeouts, prefixes, etc.) are properly restored
        - Channel initialization works correctly with preserved configurations
        """
        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        custom_config = {"custom_param": "custom_value", "timeout": 30, "key_prefix": "test_"}

        context = ExecutionContext.create(
            graph=graph, start_node="test_task", max_steps=10, channel_backend="memory", config=custom_config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify channel is functional
            assert loaded_context.channel is not None

            # Test channel functionality (memory channel should work)
            loaded_context.channel.set("config_test", "works")
            assert loaded_context.channel.get("config_test") == "works"

    def test_execution_state_preservation(self):
        """Test that execution state is fully preserved during serialization.

        This test verifies that:
        - Task execution results (both simple and complex data) are preserved
        - Step counters and execution progress tracking survive serialization
        - CycleController state (cycle counts, node-specific limits) is maintained
        - All execution metadata remains consistent after deserialization
        - Internal execution state can be accurately restored for resumption
        """
        graph = TaskGraph()

        # Use simple Task without local functions to avoid pickle issues
        simple_task = Task("sample_task", register_to_context=False)
        graph.add_node(simple_task, "sample_task")

        context = ExecutionContext.create(graph=graph, start_node="sample_task", max_steps=20, default_max_retries=3)

        # Set various execution states
        context.set_result("sample_task", "task_result")
        context.set_result("other_task", {"status": "done", "data": [1, 2, 3]})
        context.increment_step()
        context.increment_step()
        context.increment_step()

        # Set some cycle controller state
        context.cycle_controller.cycle_counts["sample_task"] = 2
        context.cycle_controller.set_node_max_cycles("sample_task", 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify execution state
            assert loaded_context.steps == 3
            assert loaded_context.get_result("sample_task") == "task_result"
            assert loaded_context.get_result("other_task") == {"status": "done", "data": [1, 2, 3]}

            # Verify cycle controller state
            assert loaded_context.cycle_controller.cycle_counts["sample_task"] == 2
            assert loaded_context.cycle_controller.get_max_cycles_for_node("sample_task") == 5

    def test_queue_state_preservation(self):
        """Ensure the in-memory queue state persists after serialization."""
        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        context = ExecutionContext.create(graph=graph, start_node="test_task", max_steps=10, channel_backend="memory")

        # Test that queue can be serialized and reconstructed
        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context_queue.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify queue functionality
            assert loaded_context.task_queue is not None

            # Queue should contain the start_node task
            next_task_id = loaded_context.task_queue.get_next_task()
            assert next_task_id == "test_task"

            # Test that queue is functional after reconstruction
            assert loaded_context.max_steps == 10

    # test_task_resolver_preservation removed - TaskResolver no longer exists
    # Tasks are now stored in Graph (via GraphStore) and retrieved directly

    def test_serialization_error_handling(self):
        """Test error handling during serialization/deserialization.

        This test verifies that:
        - Appropriate exceptions are raised for invalid file paths during save operations
        - FileNotFoundError is properly handled when loading non-existent files
        - Corrupted or invalid pickle data results in predictable exception behavior
        - Error conditions are handled gracefully without system crashes
        - Robust error handling maintains system stability during edge cases
        """
        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        context = ExecutionContext.create(graph=graph, start_node="test_task", max_steps=10)

        # Test saving to invalid path
        with pytest.raises((OSError, PermissionError, FileNotFoundError)):
            context.save("/invalid/path/context.pkl")

        # Test loading from non-existent file
        with pytest.raises(FileNotFoundError):
            ExecutionContext.load("non_existent_file.pkl")

        # Test loading corrupted file
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted_path = os.path.join(tmpdir, "corrupted.pkl")
            with open(corrupted_path, "w") as f:
                f.write("corrupted data")
            with pytest.raises((pickle.UnpicklingError, AttributeError, EOFError)):
                ExecutionContext.load(corrupted_path)
                ExecutionContext.load(corrupted_path)

    def test_lambda_task_serialization(self):
        """Test ExecutionContext serialization with lambda tasks using cloudpickle.

        This test verifies that:
        - Lambda functions can be serialized with cloudpickle
        - ExecutionContext containing lambda tasks can be pickled
        - Lambda tasks work correctly after deserialization
        - cloudpickle properly handles lambda closures
        """
        from graflow.core.serialization import dumps, loads
        from graflow.core.task import TaskWrapper

        graph = TaskGraph()

        # Create lambda task (only possible with cloudpickle)
        lambda_task = TaskWrapper("lambda_task", lambda x: x * 2)
        graph.add_node(lambda_task, "lambda_task")

        context = ExecutionContext.create(graph)

        # Serialize using cloudpickle
        pickled = dumps(context)
        restored = loads(pickled)

        # Verify task is restored
        assert "lambda_task" in restored.graph.nodes
        restored_task = restored.graph.get_node("lambda_task")

        # Verify lambda works
        result = restored_task.func(5)
        assert result == 10

    def test_closure_task_serialization(self):
        """Test ExecutionContext serialization with closure tasks using cloudpickle.

        This test verifies that:
        - Closures referencing outer scope variables can be serialized
        - ExecutionContext containing closure tasks can be pickled
        - Closure tasks maintain their captured state after deserialization
        - cloudpickle properly handles closure environments
        """
        from graflow.core.serialization import dumps, loads
        from graflow.core.task import TaskWrapper

        graph = TaskGraph()

        # Create closure
        multiplier = 3

        def create_task():
            def inner_task(x):
                return x * multiplier  # References outer scope variable

            return inner_task

        closure_task = TaskWrapper("closure_task", create_task())
        graph.add_node(closure_task, "closure_task")

        context = ExecutionContext.create(graph)

        # Serialize using cloudpickle
        pickled = dumps(context)
        restored = loads(pickled)

        # Verify task is restored
        restored_task = restored.graph.get_node("closure_task")

        # Verify closure works with captured state
        result = restored_task.func(4)
        assert result == 12  # 4 * 3

    def test_task_spec_with_lambda_serialization(self):
        """Test TaskSpec serialization with lambda ExecutionContext using cloudpickle.

        This test verifies that:
        - TaskSpec containing ExecutionContext with lambda can be serialized
        - Lambda tasks in TaskSpec work after Redis-style serialization
        - Worker processes can receive and execute lambda tasks
        - cloudpickle enables flexible task definition patterns
        """
        from graflow.core.serialization import dumps, loads
        from graflow.core.task import TaskWrapper
        from graflow.queue.base import TaskSpec

        graph = TaskGraph()

        # Create lambda task (cloudpickle allows this)
        lambda_task = TaskWrapper("test_task", lambda: 42)
        graph.add_node(lambda_task, "test_task")

        context = ExecutionContext.create(graph, "test_task")
        lambda_task.set_execution_context(context)

        # Create TaskSpec
        task_spec = TaskSpec(executable=lambda_task, execution_context=context)

        # Serialize (simulating Redis queue, using cloudpickle)
        pickled = dumps(task_spec)

        # Deserialize
        restored_spec = loads(pickled)

        # Verify ExecutionContext is restored
        assert restored_spec.execution_context is not None
        assert restored_spec.execution_context.session_id == context.session_id
        assert restored_spec.task_id == "test_task"

        # Verify lambda task works
        restored_task = restored_spec.executable
        result = restored_task.func()
        assert result == 42


class TestExecutionContextLLMSerialization:
    """Test suite for LLM client and agent serialization in ExecutionContext.

    This test class validates that LLM-related components (LLMClient and LLMAgent)
    can be properly serialized and deserialized, which is critical for:

    - Distributed task execution with LLM-powered tasks
    - Worker processes accessing shared LLM clients
    - Agent persistence across process boundaries
    - Seamless state recovery in distributed workflows
    """

    @pytest.fixture(autouse=True)
    def mock_llm_env(self, monkeypatch):
        """Mock environment variables for LLMClient."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-dummy")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-dummy")
        monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")

    def test_llm_client_explicit_serialization(self):
        """Test that explicitly set LLMClient is preserved during serialization.

        Verifies that:
        - Explicitly injected LLMClient instances survive serialization
        - Model configuration is preserved
        - Client remains functional after deserialization
        - Default parameters are maintained
        """
        pytest.importorskip("litellm")
        from graflow.llm.client import LLMClient

        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        # Create explicit LLMClient with custom configuration
        llm_client = LLMClient(model="gpt-4o", temperature=0.7, max_tokens=1000)

        context = ExecutionContext.create(graph=graph, max_steps=10, llm_client=llm_client)

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify LLMClient is preserved
            assert loaded_context._llm_client is not None
            loaded_client = loaded_context.llm_client
            assert loaded_client.model == "gpt-4o"
            assert loaded_client.default_params["temperature"] == 0.7
            assert loaded_client.default_params["max_tokens"] == 1000

    def test_llm_client_auto_creation_after_deserialization(self):
        """Test that LLMClient is auto-created on first access after deserialization.

        Verifies that:
        - Context without explicit LLMClient can be serialized
        - LLMClient is lazily created on first property access
        - Default model (gpt-5-mini) is used
        - Auto-creation works correctly in deserialized context
        """
        pytest.importorskip("litellm")

        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        # Create context WITHOUT explicit LLMClient
        context = ExecutionContext.create(graph=graph, max_steps=10)

        # Don't access llm_client property yet
        assert context._llm_client is None

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify _llm_client is None after deserialization
            assert loaded_context._llm_client is None

            # Access llm_client property - should auto-create
            client = loaded_context.llm_client

            # Verify auto-created client has default model
            assert client is not None
            assert client.model == "gpt-5-mini"

    def test_llm_agent_serialization_adk(self):
        """Test that LLMAgent (ADK) is serialized to YAML and restored.

        Verifies that:
        - AdkLLMAgent instances are serialized to YAML
        - YAML representation is preserved in _llm_agents_yaml
        - Agent instances are NOT directly serialized (memory optimization)
        - Agents can be restored from YAML in worker processes
        """
        try:
            from google.adk.agents import LlmAgent  # noqa: I001
            from graflow.llm.agents.adk_agent import AdkLLMAgent
        except ImportError:
            pytest.skip("google-adk not available")

        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        context = ExecutionContext.create(graph=graph, max_steps=10)

        # Create and register ADK agent
        adk_agent = LlmAgent(name="test_supervisor", model="gemini-2.5-flash")
        agent = AdkLLMAgent(adk_agent, app_name=context.session_id)

        context.register_llm_agent("supervisor", agent)

        # Verify agent is registered and YAML is created
        assert "supervisor" in context._llm_agents
        assert "supervisor" in context._llm_agents_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save context
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify agent instances are NOT preserved (memory optimization)
            assert len(loaded_context._llm_agents) == 0

            # Verify YAML is preserved
            assert "supervisor" in loaded_context._llm_agents_yaml

            # Access agent - should restore from YAML
            restored_agent = loaded_context.get_llm_agent("supervisor")

            # Verify agent is restored and functional
            assert restored_agent is not None
            assert isinstance(restored_agent, AdkLLMAgent)
            assert restored_agent._adk_agent.name == "test_supervisor"
            assert restored_agent._adk_agent.model == "gemini-2.5-flash"

            # Verify restored agent is cached
            assert "supervisor" in loaded_context._llm_agents

    def test_llm_agent_not_found_after_deserialization(self):
        """Test that accessing non-existent agent raises KeyError.

        Verifies that:
        - Agent registry behaves correctly after deserialization
        - Proper error is raised for missing agents
        - Error messages are informative
        """
        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        context = ExecutionContext.create(graph=graph, max_steps=10)

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load empty context
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Try to access non-existent agent
            with pytest.raises(KeyError, match="LLMAgent 'nonexistent' not found"):
                loaded_context.get_llm_agent("nonexistent")

    def test_multiple_llm_agents_serialization(self):
        """Test serialization with multiple registered agents.

        Verifies that:
        - Multiple agents can be registered and serialized
        - Each agent's YAML is preserved independently
        - All agents can be restored correctly
        - Agent isolation is maintained
        """
        try:
            from google.adk.agents import LlmAgent  # noqa: I001
            from graflow.llm.agents.adk_agent import AdkLLMAgent
        except ImportError:
            pytest.skip("google-adk not available")

        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        context = ExecutionContext.create(graph=graph, max_steps=10)

        # Register multiple agents
        agent1 = AdkLLMAgent(LlmAgent(name="agent1", model="gemini-2.5-flash"), app_name=context.session_id)
        agent2 = AdkLLMAgent(LlmAgent(name="agent2", model="gemini-2.5-flash"), app_name=context.session_id)
        agent3 = AdkLLMAgent(LlmAgent(name="agent3", model="gemini-2.5-flash"), app_name=context.session_id)

        context.register_llm_agent("agent1", agent1)
        context.register_llm_agent("agent2", agent2)
        context.register_llm_agent("agent3", agent3)

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify all agents can be restored
            restored_agent1 = loaded_context.get_llm_agent("agent1")
            restored_agent2 = loaded_context.get_llm_agent("agent2")
            restored_agent3 = loaded_context.get_llm_agent("agent3")
            assert isinstance(restored_agent1, AdkLLMAgent)
            assert isinstance(restored_agent2, AdkLLMAgent)
            assert isinstance(restored_agent3, AdkLLMAgent)

            # Verify each agent is correct
            assert restored_agent1._adk_agent.name == "agent1"
            assert restored_agent2._adk_agent.name == "agent2"
            assert restored_agent3._adk_agent.name == "agent3"

    def test_llm_client_and_agent_together(self):
        """Test serialization with both LLMClient and LLMAgent.

        Verifies that:
        - LLMClient and LLMAgent can coexist in serialized context
        - Both are preserved independently
        - No interference between client and agent serialization
        - Both remain functional after deserialization
        """
        try:
            pytest.importorskip("litellm")
            from google.adk.agents import LlmAgent  # noqa: I001
            from graflow.llm.agents.adk_agent import AdkLLMAgent
            from graflow.llm.client import LLMClient
        except ImportError:
            pytest.skip("litellm or google-adk not available")

        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        # Create context with LLMClient
        llm_client = LLMClient(model="gpt-4o", temperature=0.5)

        context = ExecutionContext.create(graph=graph, max_steps=10, llm_client=llm_client)

        # Register agent
        adk_agent = LlmAgent(name="supervisor", model="gemini-2.5-flash")
        agent = AdkLLMAgent(adk_agent, app_name=context.session_id)
        context.register_llm_agent("supervisor", agent)

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify LLMClient is preserved
            loaded_client = loaded_context.llm_client
            assert loaded_client.model == "gpt-4o"
            assert loaded_client.default_params["temperature"] == 0.5

            # Verify agent can be restored
            restored_agent = loaded_context.get_llm_agent("supervisor")
            assert isinstance(restored_agent, AdkLLMAgent)
            assert restored_agent._adk_agent.name == "supervisor"

    def test_backward_compatibility_without_llm_attributes(self):
        """Test that old checkpoints without LLM attributes can be loaded.

        Verifies that:
        - __setstate__ initializes missing LLM attributes
        - Backward compatibility is maintained
        - Old checkpoints work with new code
        - Default values are applied correctly
        """
        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        context = ExecutionContext.create(graph=graph, max_steps=10)

        # Manually create a "legacy" state without LLM attributes
        state = context.__getstate__()

        # Remove LLM attributes to simulate old checkpoint
        state.pop("_llm_client", None)
        state.pop("_llm_agents", None)
        state.pop("_llm_agents_yaml", None)

        # Create new context and restore legacy state
        new_context = ExecutionContext.__new__(ExecutionContext)
        new_context.__setstate__(state)

        # Verify LLM attributes are initialized
        assert hasattr(new_context, "_llm_client")
        assert hasattr(new_context, "_llm_agents")
        assert hasattr(new_context, "_llm_agents_yaml")

        assert new_context._llm_client is None
        assert new_context._llm_agents == {}
        assert new_context._llm_agents_yaml == {}

        # Verify auto-creation still works
        pytest.importorskip("litellm")
        client = new_context.llm_client
        assert client.model == "gpt-5-mini"

    def test_llm_client_serialization(self):
        """Test that ExecutionContext with llm_client survives pickle.

        This is a simple, focused test verifying that:
        - LLMClient can be pickled directly
        - ExecutionContext with LLMClient survives pickle operations
        - Client model and configuration remain intact
        - No data loss during serialization/deserialization
        """
        pytest.importorskip("litellm")
        from graflow.llm.client import LLMClient

        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        # Create LLMClient with specific configuration
        llm_client = LLMClient(model="claude-3-5-sonnet-20241022", temperature=0.3)

        # Create context with LLMClient
        context = ExecutionContext.create(graph=graph, start_node="test_task", max_steps=10, llm_client=llm_client)

        # Pickle and unpickle the entire context
        pickled = pickle.dumps(context)
        restored_context = pickle.loads(pickled)

        # Verify LLMClient survived pickle
        assert restored_context._llm_client is not None
        restored_client = restored_context.llm_client

        # Verify configuration is preserved
        assert restored_client.model == "claude-3-5-sonnet-20241022"
        assert restored_client.default_params["temperature"] == 0.3

        # Verify client is functional (basic check)
        assert isinstance(restored_client, LLMClient)

    def test_llm_agent_yaml_roundtrip(self):
        """Test ADK agent can be serialized to YAML and back.

        This test verifies the complete YAML roundtrip for ADK LlmAgent:
        - Agent can be serialized to YAML string using agent_to_yaml()
        - YAML string preserves agent configuration
        - Agent can be reconstructed from YAML using yaml_to_agent()
        - Reconstructed agent maintains original configuration
        - No information loss during YAML roundtrip
        """
        try:
            from google.adk.agents import LlmAgent

            from graflow.llm.serialization import agent_to_yaml, yaml_to_agent
        except ImportError:
            pytest.skip("google-adk not available")

        # Create original ADK agent
        original_agent = LlmAgent(name="yaml_test_agent", model="gemini-2.5-flash")

        # Serialize to YAML
        yaml_str = agent_to_yaml(original_agent)
        assert yaml_str is not None
        assert isinstance(yaml_str, str)
        assert "yaml_test_agent" in yaml_str
        assert "gemini-2.5-flash" in yaml_str

        # Deserialize from YAML (roundtrip)
        restored_agent = yaml_to_agent(yaml_str)

        # Verify agent is properly reconstructed
        assert restored_agent is not None
        assert isinstance(restored_agent, LlmAgent)
        assert restored_agent.name == "yaml_test_agent"
        assert restored_agent.model == "gemini-2.5-flash"

        # Verify it's a new instance (not the same object)
        assert restored_agent is not original_agent

        # Verify second roundtrip produces same result
        yaml_str2 = agent_to_yaml(restored_agent)
        restored_agent2 = yaml_to_agent(yaml_str2)
        assert restored_agent2.name == "yaml_test_agent"
        assert restored_agent2.model == "gemini-2.5-flash"
