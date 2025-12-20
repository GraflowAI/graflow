"""Tests for task instance binding functionality.

This module tests the ability to create multiple task instances from a single
@task decorated function with bound parameters.
"""

import pytest

from graflow.core.decorators import task
from graflow.core.workflow import workflow


class TestInstanceCreation:
    """Test creating multiple instances from same @task function."""

    def test_multiple_instances_with_unique_task_ids(self):
        """Test that multiple instances can be created with unique task_ids."""
        @task
        def process_data(value: int) -> int:
            return value * 2

        # Create multiple instances with different task_ids
        task1 = process_data(task_id="task1", value=10)
        task2 = process_data(task_id="task2", value=20)

        # Verify they are different instances
        assert task1 is not task2
        assert task1.task_id == "task1"
        assert task2.task_id == "task2"

        # Verify bound parameters are stored
        assert task1._bound_kwargs == {"value": 10}
        assert task2._bound_kwargs == {"value": 20}

    def test_instance_creation_with_only_task_id(self):
        """Test creating instance with only task_id, no bound parameters."""
        @task
        def process_data(value: int) -> int:
            return value * 2

        task_instance = process_data(task_id="my_task")

        assert task_instance.task_id == "my_task"
        assert task_instance._bound_kwargs == {}

    def test_auto_generated_task_id(self):
        """Test auto-generation of task_id when not specified."""
        @task
        def process_data(value: int) -> int:
            return value * 2

        # Create instance without task_id
        task1 = process_data(value=10)
        task2 = process_data(value=20)

        # Verify task_ids are auto-generated with correct format
        assert task1.task_id.startswith("process_data_")
        assert task2.task_id.startswith("process_data_")

        # Verify they are unique
        assert task1.task_id != task2.task_id

        # Verify format: {func_name}_{8-char-hash}
        task_id_parts = task1.task_id.split("_")
        assert len(task_id_parts) == 3  # process, data, {hash}
        assert len(task_id_parts[2]) == 8  # 8-character hash

    def test_positional_args_not_supported(self):
        """Test that positional arguments raise TypeError."""
        @task
        def process_data(value: int) -> int:
            return value * 2

        # Should raise TypeError for positional args
        with pytest.raises(TypeError, match="does not support positional arguments"):
            process_data("task_id", 10)

    def test_instance_creation_preserves_task_attributes(self):
        """Test that new instances preserve task configuration."""
        @task(inject_context=True)
        def process_data(ctx, value: int) -> int:
            return value * 2

        # Create instance with bound parameters
        task_instance = process_data(task_id="test", value=10)

        # Verify task attributes are preserved
        assert task_instance.inject_context is True
        assert task_instance.resolve_keyword_args is True
        assert task_instance._bound_kwargs == {"value": 10}


class TestParameterResolution:
    """Test parameter resolution priority: channel < bound < injection."""

    def test_bound_params_override_channel(self):
        """Test that bound parameters override channel values."""
        from graflow.core.context import TaskExecutionContext

        @task(inject_context=True)
        def setup(ctx: TaskExecutionContext) -> None:
            # Set channel values
            channel = ctx.get_channel()
            channel.set("value", 5)
            channel.set("multiplier", 2)

        @task
        def process(value: int, multiplier: int = 1) -> int:
            return value * multiplier

        with workflow("test") as wf:
            # Create task with bound parameter that should override channel
            task_instance = process(task_id="test", value=10)

            # Execute
            _ = setup >> task_instance
            _, exec_context = wf.execute(setup.task_id, ret_context=True)

        # Bound value (10) should override channel value (5)
        # Multiplier should come from channel (2)
        assert exec_context.get_result("test") == 20  # 10 * 2

    def test_channel_provides_fallback_for_unbound_params(self):
        """Test that channel provides values for parameters not bound."""
        from graflow.core.context import TaskExecutionContext

        @task(inject_context=True)
        def setup(ctx: TaskExecutionContext) -> None:
            # Set channel values
            channel = ctx.get_channel()
            channel.set("value", 5)
            channel.set("multiplier", 3)

        @task
        def process(value: int, multiplier: int) -> int:
            return value * multiplier

        with workflow("test") as wf:
            # Create task with only one bound parameter
            task_instance = process(task_id="test", value=10)

            # Execute
            _ = setup >> task_instance
            _, exec_context = wf.execute(setup.task_id, ret_context=True)

        # value from bound (10), multiplier from channel (3)
        assert exec_context.get_result("test") == 30

    def test_priority_order_channel_bound_injection(self):
        """Test full priority order: channel < bound < injection."""
        from graflow.core.context import TaskExecutionContext

        @task(inject_context=True)
        def setup(ctx: TaskExecutionContext) -> None:
            # Set channel value (lowest priority)
            channel = ctx.get_channel()
            channel.set("value", 5)

        @task(inject_context=True)
        def process(ctx: TaskExecutionContext, value: int) -> dict:
            return {
                "value": value,
                "has_context": ctx is not None
            }

        with workflow("test") as wf:
            # Create task with bound parameter (overrides channel)
            task_instance = process(task_id="test", value=10)

            # Execute (context injection has highest priority)
            _ = setup >> task_instance
            _, exec_context = wf.execute(setup.task_id, ret_context=True)

        # Bound value overrides channel, context is injected
        result = exec_context.get_result("test")
        assert result["value"] == 10
        assert result["has_context"] is True


class TestRegistration:
    """Test task registration behavior."""

    def test_immediate_registration_with_context(self):
        """Test that tasks are registered immediately when context exists."""
        with workflow("test") as wf:
            @task
            def process_data(value: int) -> int:
                return value * 2

            _task1 = process_data(task_id="task1", value=10)
            _task2 = process_data(task_id="task2", value=20)

            # Tasks should be registered to graph
            assert "task1" in wf.graph.nodes
            assert "task2" in wf.graph.nodes

    def test_lazy_registration_without_context(self):
        """Test lazy registration when no context exists."""
        @task
        def process_data(value: int) -> int:
            return value * 2

        # Create instance outside of workflow context
        task_instance = process_data(task_id="test", value=10)

        # Should have pending registration flag
        assert task_instance._pending_registration is True

        # Register when added to workflow via operator
        with workflow("test") as wf:
            @task
            def dummy() -> None:
                pass

            # Use >> operator to trigger registration
            _ = dummy >> task_instance
            assert "test" in wf.graph.nodes
            assert task_instance._pending_registration is False


class TestSerialization:
    """Test that bound parameters are stored correctly for serialization."""

    def test_bound_kwargs_stored_correctly(self):
        """Test that bound kwargs are stored in the instance."""
        @task
        def process_data(value: int, multiplier: int = 2) -> int:
            return value * multiplier

        # Create instance with bound parameters
        task_instance = process_data(task_id="test", value=10, multiplier=3)

        # Verify bound kwargs are stored
        assert hasattr(task_instance, '_bound_kwargs')
        assert task_instance._bound_kwargs == {"value": 10, "multiplier": 3}
        assert task_instance.task_id == "test"

    def test_bound_kwargs_included_in_state(self):
        """Test that bound kwargs are included in __getstate__."""
        @task
        def process_data(value: int) -> int:
            return value * 2

        task_instance = process_data(task_id="test", value=10)

        # Get state for serialization
        state = task_instance.__getstate__()

        # Verify bound kwargs are in state
        assert '_bound_kwargs' in state
        assert state['_bound_kwargs'] == {"value": 10}


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_bound_kwargs(self):
        """Test task with no bound parameters."""
        @task
        def process_data(value: int) -> int:
            return value * 2

        task_instance = process_data(task_id="test")

        assert task_instance._bound_kwargs == {}
        assert task_instance.task_id == "test"

    def test_execution_mode_with_user_kwargs(self):
        """Test that execution mode still allows user-provided kwargs."""
        result_holder = []

        @task
        def process(value: int, multiplier: int = 1) -> int:
            result = value * multiplier
            result_holder.append(result)
            return result

        with workflow("test") as wf:
            task_instance = process(task_id="test", value=10)

            # Execute workflow with bound value
            _, exec_context = wf.execute(task_instance.task_id, ret_context=True)

        # Bound value should be used
        assert exec_context.get_result("test") == 10  # 10 * 1 (default multiplier)

    def test_multiple_bound_params(self):
        """Test binding multiple parameters."""
        @task
        def process(a: int, b: int, c: int, d: int = 4) -> int:
            return a + b + c + d

        with workflow("test") as wf:
            task_instance = process(task_id="test", a=1, b=2, c=3, d=10)

            assert task_instance._bound_kwargs == {"a": 1, "b": 2, "c": 3, "d": 10}

            _, exec_context = wf.execute(task_instance.task_id, ret_context=True)

        assert exec_context.get_result("test") == 16  # 1 + 2 + 3 + 10
