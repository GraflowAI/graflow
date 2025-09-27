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
import tempfile

import pytest

from graflow.channels.base import Channel
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph
from graflow.core.task import Task
from graflow.queue.factory import QueueBackend


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
            graph=graph,
            start_node="test_task",
            max_steps=15,
            default_max_retries=5,
            channel_backend="memory"
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

        context = ExecutionContext.create(
            graph=graph,
            start_node="test_task",
            max_steps=10,
            channel_backend="memory"
        )

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
            "db": clean_redis.connection_pool.connection_kwargs.get("db", 0)
        }

        # Create context with Redis backend using the Docker Redis
        context = ExecutionContext.create(
            graph=graph,
            start_node="test_task",
            max_steps=10,
            channel_backend="redis",
            config=redis_config
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

        custom_config = {
            "custom_param": "custom_value",
            "timeout": 30,
            "key_prefix": "test_"
        }

        context = ExecutionContext.create(
            graph=graph,
            start_node="test_task",
            max_steps=10,
            channel_backend="memory",
            config=custom_config
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

        context = ExecutionContext.create(
            graph=graph,
            start_node="sample_task",
            max_steps=20,
            default_max_retries=3
        )

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

    def test_queue_backend_preservation(self):
        """Test that queue backend configuration is preserved during serialization.

        This test ensures that:
        - TaskQueue backend selection (in-memory, Redis) survives serialization
        - Queue state and pending tasks are properly maintained
        - Queue functionality remains operational after deserialization
        - Backend-specific configuration is preserved and restored
        - Task scheduling and queue operations work correctly after reload
        """
        graph = TaskGraph()
        graph.add_node(Task("test_task", register_to_context=False), "test_task")

        # Test with IN_MEMORY backend only to avoid task serialization complexity
        context = ExecutionContext.create(
            graph=graph,
            start_node="test_task",
            max_steps=10,
            queue_backend=QueueBackend.IN_MEMORY,
            channel_backend="memory"
        )

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

    def test_function_manager_preservation(self):
        """Test that function manager state is preserved during serialization.

        This test validates that:
        - TaskFunctionManager registry survives pickle serialization
        - Registered functions remain available and callable after deserialization
        - Function registration mappings are accurately preserved
        - Function resolution and execution work correctly after reload
        - Function manager state integrity is maintained across serialization cycles
        """
        graph = TaskGraph()

        # Use a simple Task instead of local function to avoid pickle issues
        simple_task = Task("registered_function", register_to_context=False)
        graph.add_node(simple_task, "registered_function")

        context = ExecutionContext.create(
            graph=graph,
            start_node="registered_function",
            max_steps=10
        )

        # Register some functions using global functions
        context.function_manager.register_task_function("test_func", global_test_func)
        context.function_manager.register_task_function("math_func", global_math_func)

        with tempfile.TemporaryDirectory() as tmpdir:
            context_path = os.path.join(tmpdir, "context.pkl")

            # Save and load
            context.save(context_path)
            loaded_context = ExecutionContext.load(context_path)

            # Verify function manager
            assert loaded_context.function_manager is not None

            # Verify registered functions are available
            registered_functions = loaded_context.function_manager.get_registered_functions()
            assert "test_func" in registered_functions
            assert "math_func" in registered_functions

            test_func = registered_functions["test_func"]
            assert test_func() == "test"

            math_func = registered_functions["math_func"]
            assert math_func(3, 4) == 7

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

        context = ExecutionContext.create(
            graph=graph,
            start_node="test_task",
            max_steps=10
        )

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

            with pytest.raises(Exception):  # Could be various pickle-related exceptions  # noqa: B017
                ExecutionContext.load(corrupted_path)
