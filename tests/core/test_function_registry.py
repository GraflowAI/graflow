"""Comprehensive unit tests for function registry and serialization."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.function_registry import (
    FunctionRegistry,
    FunctionResolutionError,
    FunctionSerializer,
    TaskFunctionManager,
    default_function_manager,
    register_task_function,
    resolve_function,
    serialize_function,
)
from graflow.core.graph import TaskGraph
from graflow.core.task import ParallelGroup, Task, TaskWrapper


class TestFunctionRegistry:
    """Test cases for FunctionRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        FunctionRegistry.clear()

    def test_register_and_get_function(self):
        """Test basic function registration and retrieval."""
        def test_func():
            return "test_result"

        # Register function
        FunctionRegistry.register("test_func", test_func)

        # Retrieve function
        retrieved_func = FunctionRegistry.get("test_func")
        assert retrieved_func is test_func
        assert retrieved_func() == "test_result" # type: ignore

    def test_get_nonexistent_function(self):
        """Test getting non-existent function returns None."""
        result = FunctionRegistry.get("nonexistent")
        assert result is None

    def test_list_functions(self):
        """Test listing registered functions."""
        def func1():
            pass
        def func2():
            pass

        FunctionRegistry.register("func1", func1)
        FunctionRegistry.register("func2", func2)

        functions = FunctionRegistry.list_functions()
        assert set(functions) == {"func1", "func2"}

    def test_clear_registry(self):
        """Test clearing the registry."""
        def test_func():
            pass

        FunctionRegistry.register("test", test_func)
        assert len(FunctionRegistry.list_functions()) == 1

        FunctionRegistry.clear()
        assert len(FunctionRegistry.list_functions()) == 0


class TestFunctionSerializer:
    """Test cases for FunctionSerializer class."""

    def test_serialize_reference_strategy(self):
        """Test reference serialization strategy."""
        def test_function():
            return "reference_test"

        result = FunctionSerializer.serialize_function(test_function, "reference")

        expected = {
            "strategy": "reference",
            "module": test_function.__module__,
            "name": "test_function",
            "qualname": "TestFunctionSerializer.test_serialize_reference_strategy.<locals>.test_function"
        }

        assert result == expected

    def test_serialize_pickle_strategy(self):
        """Test pickle serialization strategy."""
        def test_function():
            return "pickle_test"

        result = FunctionSerializer.serialize_function(test_function, "pickle")

        assert result["strategy"] == "pickle"
        assert result["name"] == "test_function"
        assert result["module"] == test_function.__module__
        assert "data" in result

        # Verify data can be decoded
        pickled_data = base64.b64decode(result["data"].encode('utf-8'))
        assert pickled_data is not None

    def test_serialize_source_strategy(self):
        """Test source code serialization strategy."""
        def test_function():
            return "source_test"

        result = FunctionSerializer.serialize_function(test_function, "source")

        assert result["strategy"] == "source"
        assert result["name"] == "test_function"
        assert result["module"] == test_function.__module__
        assert "source" in result
        assert "def test_function():" in result["source"]

    def test_serialize_invalid_strategy(self):
        """Test that invalid strategy raises ValueError."""
        def test_func():
            pass

        with pytest.raises(ValueError, match="Unsupported serialization strategy"):
            FunctionSerializer.serialize_function(test_func, "invalid_strategy")

    def test_deserialize_reference_success(self):
        """Test successful reference deserialization."""
        # Use a built-in function that's always available
        func_data = {
            "strategy": "reference",
            "module": "builtins",
            "name": "len"
        }

        func, method = FunctionSerializer.deserialize_function(func_data)
        assert func is len
        assert method == "import"
        assert func([1, 2, 3]) == 3

    def test_deserialize_pickle_success(self):
        """Test successful pickle deserialization."""
        def original_function():
            return "pickled_result"

        # Serialize first
        serialized = FunctionSerializer.serialize_function(original_function, "pickle")

        # Then deserialize
        func, method = FunctionSerializer.deserialize_function(serialized)
        assert method == "pickle"
        assert func() == "pickled_result"

    def test_deserialize_source_success(self):
        """Test successful source code deserialization."""
        source_code = '''def test_source_func():
    return "source_result"'''

        func_data = {
            "strategy": "source",
            "source": source_code,
            "name": "test_source_func",
            "module": "test_module"
        }

        func, method = FunctionSerializer.deserialize_function(func_data)
        assert method == "source"
        assert func() == "source_result"

    def test_deserialize_with_registry_fallback(self):
        """Test registry fallback when primary strategy fails."""
        # Register a function in registry
        def fallback_function():
            return "fallback_result"

        FunctionRegistry.register("test_fallback", fallback_function)

        # Create func_data that will fail primary strategy but succeed with registry
        func_data = {
            "strategy": "reference",
            "module": "nonexistent_module",
            "name": "test_fallback"
        }

        func, method = FunctionSerializer.deserialize_function(func_data)
        assert func is fallback_function
        assert method == "registry"
        assert func() == "fallback_result"

    def test_deserialize_with_module_name_fallback(self):
        """Test registry fallback with module.name pattern."""
        def module_function():
            return "module_result"

        FunctionRegistry.register("test.module.module_function", module_function)

        func_data = {
            "strategy": "reference",
            "module": "test.module",
            "name": "module_function"
        }

        func, method = FunctionSerializer.deserialize_function(func_data)
        assert func is module_function
        assert method == "registry"

    def test_deserialize_complete_failure(self):
        """Test that complete failure raises FunctionResolutionError."""
        func_data = {
            "strategy": "reference",
            "module": "completely_nonexistent_module",
            "name": "nonexistent_function"
        }

        with pytest.raises(FunctionResolutionError, match="Cannot resolve function"):
            FunctionSerializer.deserialize_function(func_data)

    def test_try_deserialize_function_success(self):
        """Test safe wrapper returns function on success."""
        func_data = {
            "strategy": "reference",
            "module": "builtins",
            "name": "abs"
        }

        func, method = FunctionSerializer.try_deserialize_function(func_data)
        assert func is abs
        assert method == "import"

    def test_try_deserialize_function_failure(self):
        """Test safe wrapper returns None on failure."""
        func_data = {
            "strategy": "reference",
            "module": "nonexistent",
            "name": "nonexistent"
        }

        func, method = FunctionSerializer.try_deserialize_function(func_data)
        assert func is None
        assert method == "failed"

    def test_deserialize_source_with_no_callable(self):
        """Test source deserialization failure when no callable found."""
        func_data = {
            "strategy": "source",
            "source": "x = 42",  # No function definition
            "name": "test_func",
            "module": "test_module"
        }

        # Since fallback to registry happens, expect the final error message
        with pytest.raises(FunctionResolutionError, match="Cannot resolve function"):
            FunctionSerializer.deserialize_function(func_data)

    def test_deserialize_source_with_syntax_error(self):
        """Test source deserialization failure with invalid syntax."""
        func_data = {
            "strategy": "source",
            "source": "def invalid syntax:",  # Invalid syntax
            "name": "test_func",
            "module": "test_module"
        }

        # Since fallback to registry happens, expect the final error message
        with pytest.raises(FunctionResolutionError, match="Cannot resolve function"):
            FunctionSerializer.deserialize_function(func_data)

    def setup_method(self):
        """Clear registry before each test."""
        FunctionRegistry.clear()


class TestTaskFunctionManager:
    """Test cases for TaskFunctionManager class."""

    def setup_method(self):
        """Clear registry and create fresh manager before each test."""
        FunctionRegistry.clear()
        self.manager = TaskFunctionManager(default_strategy="reference")

    def test_initialization(self):
        """Test TaskFunctionManager initialization."""
        manager = TaskFunctionManager("pickle")
        assert manager.default_strategy == "pickle"
        assert hasattr(manager, "serializer")

    def test_register_task_function(self):
        """Test task function registration."""
        def test_task():
            return "task_result"

        self.manager.register_task_function("my_task", test_task)

        # Should be registered with task_id
        assert FunctionRegistry.get("my_task") is test_task

        # Should also be registered with module.name pattern
        expected_key = f"{test_task.__module__}.{test_task.__name__}"
        assert FunctionRegistry.get(expected_key) is test_task

    def test_serialize_task_function_default_strategy(self):
        """Test serialization with default strategy."""
        def test_func():
            pass

        result = self.manager.serialize_task_function(test_func)
        assert result["strategy"] == "reference"  # Default strategy

    def test_serialize_task_function_override_strategy(self):
        """Test serialization with overridden strategy."""
        def test_func():
            pass

        result = self.manager.serialize_task_function(test_func, "pickle")
        assert result["strategy"] == "pickle"

    def test_resolve_task_function(self):
        """Test task function resolution."""
        func_data = {
            "strategy": "reference",
            "module": "builtins",
            "name": "str"
        }

        resolved_func = self.manager.resolve_task_function(func_data)
        assert resolved_func is str

    def test_get_registered_functions(self):
        """Test getting all registered functions."""
        def func1():
            pass
        def func2():
            pass

        self.manager.register_task_function("task1", func1)
        self.manager.register_task_function("task2", func2)

        registered = self.manager.get_registered_functions()
        assert "task1" in registered
        assert "task2" in registered
        assert registered["task1"] is func1
        assert registered["task2"] is func2


class TestExecutableTypeSerialization:
    """Test serialization of different Executable types."""

    def setup_method(self):
        """Set up test environment."""
        FunctionRegistry.clear()
        self.graph = TaskGraph()
        self.context = ExecutionContext(self.graph, start_node="test")
        self.manager = TaskFunctionManager()

    def test_task_serialization(self):
        """Test Task executable serialization."""
        task = Task("test_task", register_to_context=False)

        # Task.run method should be serializable
        func_data = self.manager.serialize_task_function(task.run, "pickle")

        assert func_data["strategy"] == "pickle"
        assert func_data["name"] == "run"

        # Should be able to deserialize
        resolved_func = self.manager.resolve_task_function(func_data)
        assert callable(resolved_func)

    def test_task_wrapper_serialization(self):
        """Test TaskWrapper executable serialization."""
        def original_function():
            return "wrapper_result"

        task_wrapper = TaskWrapper("test_wrapper", original_function)

        # TaskWrapper.func should be serializable
        func_data = self.manager.serialize_task_function(task_wrapper.func, "pickle")

        assert func_data["strategy"] == "pickle"
        assert func_data["name"] == "original_function"

        # Should be able to deserialize and execute
        resolved_func = self.manager.resolve_task_function(func_data)
        assert resolved_func() == "wrapper_result"

    def test_task_decorator_serialization(self):
        """Test @task decorated function serialization."""
        @task
        def decorated_task():
            return "decorated_result"

        # Should be a TaskWrapper instance
        assert isinstance(decorated_task, TaskWrapper)

        # Serialize the wrapped function
        func_data = self.manager.serialize_task_function(decorated_task.func, "pickle")

        assert func_data["strategy"] == "pickle"
        assert func_data["name"] == "decorated_task"

        # Deserialize and verify
        resolved_func = self.manager.resolve_task_function(func_data)
        assert resolved_func() == "decorated_result"

    def test_task_decorator_with_context_serialization(self):
        """Test @task decorated function with context injection."""
        @task(inject_context=True)
        def context_task(task_ctx):
            return f"context_task_{task_ctx.task_id}"

        # Serialize the wrapped function
        func_data = self.manager.serialize_task_function(context_task.func, "pickle")

        assert func_data["strategy"] == "pickle"

        # Deserialize
        resolved_func = self.manager.resolve_task_function(func_data)

        # Test with mock context
        mock_context = MagicMock()
        mock_context.task_id = "mock_task"

        result = resolved_func(mock_context)
        assert result == "context_task_mock_task"

    def test_parallel_group_serialization(self):
        """Test ParallelGroup executable serialization."""
        task1 = Task("task1", register_to_context=False)
        task2 = Task("task2", register_to_context=False)

        # Create parallel group without registration to avoid context issues
        parallel_group = ParallelGroup.__new__(ParallelGroup)
        parallel_group._task_id = "test_group"
        parallel_group.tasks = [task1, task2]

        # Serialize the run method
        func_data = self.manager.serialize_task_function(parallel_group.run, "pickle")

        assert func_data["strategy"] == "pickle"
        assert func_data["name"] == "run"

        # Should be deserializable
        resolved_func = self.manager.resolve_task_function(func_data)
        assert callable(resolved_func)

    def test_lambda_function_serialization(self):
        """Test lambda function serialization (should work with pickle)."""
        lambda_func = lambda x: x * 2

        # Lambda should be serializable with pickle strategy
        func_data = self.manager.serialize_task_function(lambda_func, "pickle")

        assert func_data["strategy"] == "pickle"
        assert func_data["name"] == "<lambda>"

        # Deserialize and test
        resolved_func = self.manager.resolve_task_function(func_data)
        assert resolved_func(5) == 10

    def test_closure_function_serialization(self):
        """Test closure function serialization."""
        def create_closure(multiplier):
            def closure_func(x):
                return x * multiplier
            return closure_func

        closure = create_closure(3)

        # Closure should be serializable with pickle
        func_data = self.manager.serialize_task_function(closure, "pickle")

        assert func_data["strategy"] == "pickle"

        # Deserialize and test
        resolved_func = self.manager.resolve_task_function(func_data)
        assert resolved_func(4) == 12

    def test_method_serialization(self):
        """Test method serialization."""
        class TestClass:
            def __init__(self, value):
                self.value = value

            def test_method(self):
                return f"method_result_{self.value}"

        instance = TestClass("test")

        # Method should be serializable with pickle
        func_data = self.manager.serialize_task_function(instance.test_method, "pickle")

        assert func_data["strategy"] == "pickle"
        assert func_data["name"] == "test_method"

        # Deserialize and test
        resolved_func = self.manager.resolve_task_function(func_data)
        assert resolved_func() == "method_result_test"


class TestGlobalConvenienceFunctions:
    """Test global convenience functions."""

    def setup_method(self):
        """Clear registry before each test."""
        FunctionRegistry.clear()

    def test_register_task_function_global(self):
        """Test global register_task_function."""
        def global_func():
            return "global_result"

        register_task_function("global_task", global_func)

        # Should be registered in the default manager
        registered = default_function_manager.get_registered_functions()
        assert "global_task" in registered
        assert registered["global_task"] is global_func

    def test_serialize_function_global(self):
        """Test global serialize_function."""
        def test_func():
            pass

        result = serialize_function(test_func, "reference")

        assert result["strategy"] == "reference"
        assert result["name"] == "test_func"

    def test_resolve_function_global(self):
        """Test global resolve_function."""
        func_data = {
            "strategy": "reference",
            "module": "builtins",
            "name": "int"
        }

        resolved = resolve_function(func_data)
        assert resolved is int


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def setup_method(self):
        """Set up integration test environment."""
        FunctionRegistry.clear()
        self.graph = TaskGraph()
        self.context = ExecutionContext(self.graph, start_node="start")
        self.manager = TaskFunctionManager()

    def test_end_to_end_task_serialization(self):
        """Test complete end-to-end task serialization workflow."""
        # Define a complex task function
        def complex_task(data):
            result = []
            for item in data:
                if isinstance(item, (int, float)):
                    result.append(item * 2)
                else:
                    result.append(str(item).upper())
            return result

        # Create TaskWrapper
        task_wrapper = TaskWrapper("complex_task", complex_task)

        # Register for fallback resolution
        self.manager.register_task_function("complex_task", complex_task)

        # Test all serialization strategies
        strategies = ["reference", "pickle", "source"]
        test_data = [1, 2.5, "hello", 4]
        expected_result = [2, 5.0, "HELLO", 8]

        for strategy in strategies:
            with patch.object(self.manager.serializer, '_serialize_reference') as mock_ref:
                if strategy == "reference":
                    # Mock reference serialization to fail, forcing registry fallback
                    mock_ref.side_effect = ImportError("Mock import error")

                # Serialize
                func_data = self.manager.serialize_task_function(task_wrapper.func, strategy)

                # Deserialize
                resolved_func = self.manager.resolve_task_function(func_data)

                # Test execution
                result = resolved_func(test_data)
                assert result == expected_result, f"Strategy {strategy} failed"

    def test_distributed_execution_simulation(self):
        """Simulate distributed execution scenario."""
        # Worker 1: Define and serialize task
        def worker_task(x, y):
            return x + y + 100

        task_wrapper = TaskWrapper("worker_task", worker_task)
        serialized = self.manager.serialize_task_function(task_wrapper.func, "pickle")

        # Simulate sending over network (convert to/from JSON)
        import json
        json_data = json.dumps(serialized)
        received_data = json.loads(json_data)

        # Worker 2: Deserialize and execute
        resolved_func = self.manager.resolve_task_function(received_data)
        result = resolved_func(10, 20)

        assert result == 130

    def test_fallback_resolution_chain(self):
        """Test complete fallback resolution chain."""
        def fallback_function():
            return "fallback_success"

        # Register function for fallback
        FunctionRegistry.register("test.module.fallback_func", fallback_function)

        # Create func_data that will fail primary strategy
        func_data = {
            "strategy": "reference",
            "module": "test.module",
            "name": "fallback_func",
            "qualname": "fallback_func"
        }

        # Should succeed via registry fallback
        resolved, method = FunctionSerializer.deserialize_function(func_data)
        assert resolved is fallback_function
        assert method == "registry"
        assert resolved() == "fallback_success"

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        # Test function that can't be serialized with source strategy
        def non_inspectable_function():
            return "result"

        # Mock inspect.getsource to fail
        with patch('inspect.getsource', side_effect=OSError("No source available")):
            with pytest.raises(FunctionResolutionError, match="Failed to get source"):
                self.manager.serialize_task_function(non_inspectable_function, "source")

        # But should work with pickle
        result = self.manager.serialize_task_function(non_inspectable_function, "pickle")
        assert result["strategy"] == "pickle"

        # And deserialization should work
        resolved = self.manager.resolve_task_function(result)
        assert resolved() == "result"
