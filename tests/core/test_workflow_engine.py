"""Tests for WorkflowEngine with handler system."""

from unittest.mock import Mock

import pytest

from graflow import exceptions
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.core.handler import TaskHandler
from graflow.core.handlers.direct import DirectTaskHandler


class TestWorkflowEngine:
    """Test WorkflowEngine with handler registry."""

    def test_engine_initialization(self):
        """Test WorkflowEngine initializes with default handlers."""
        engine = WorkflowEngine()

        # Check that handlers dict exists
        assert hasattr(engine, "_handlers")
        assert "direct" in engine._handlers
        assert isinstance(engine._handlers["direct"], DirectTaskHandler)

    def test_register_handler(self):
        """Test registering custom handler."""
        engine = WorkflowEngine()

        # Create mock handler
        mock_handler = Mock(spec=TaskHandler)

        # Register handler
        engine.register_handler("custom", mock_handler)

        # Verify registration
        assert "custom" in engine._handlers
        assert engine._handlers["custom"] is mock_handler

    def test_get_handler_default(self):
        """Test getting default handler for task without handler_type."""
        engine = WorkflowEngine()

        # Create task without explicit handler_type
        mock_task = Mock()
        mock_task.handler_type = "direct"

        handler = engine._get_handler(mock_task)

        assert isinstance(handler, DirectTaskHandler)

    def test_get_handler_unknown_type(self):
        """Test error when requesting unknown handler type."""
        engine = WorkflowEngine()

        # Create task with unknown handler type
        mock_task = Mock()
        mock_task.handler_type = "nonexistent"

        with pytest.raises(ValueError, match="Unknown handler type: nonexistent"):
            engine._get_handler(mock_task)

    def test_execute_task_with_direct_handler(self):
        """Test executing task with direct handler."""
        # Create graph and context
        graph = TaskGraph()

        @task
        def simple_task():
            return "test_result"

        # Add task to graph
        graph.add_node(simple_task, "simple_task")

        # Create context
        context = ExecutionContext.create(graph)
        simple_task.set_execution_context(context)

        # Add task to queue
        context.add_to_queue(simple_task)

        # Execute
        engine = WorkflowEngine()
        engine.execute(context)

        # Verify result
        result = context.get_result("simple_task")
        assert result == "test_result"
        assert simple_task.handler_type == "direct"

    def test_execute_task_with_custom_handler_type(self):
        """Test task with custom handler_type attribute."""
        # Create custom mock handler
        mock_handler = Mock(spec=TaskHandler)

        # Create engine and register handler
        engine = WorkflowEngine()
        engine.register_handler("custom", mock_handler)

        # Create graph and context
        graph = TaskGraph()

        @task(handler="custom")
        def custom_task():
            return "custom_result"

        # Add task to graph
        graph.add_node(custom_task, "custom_task")

        # Create context
        context = ExecutionContext.create(graph)
        custom_task.set_execution_context(context)

        # Add task to queue
        context.add_to_queue(custom_task)

        # Execute
        engine._execute_task(custom_task, context)

        # Verify mock handler was called
        mock_handler.execute_task.assert_called_once_with(custom_task, context)
        assert custom_task.handler_type == "custom"

    def test_execute_workflow_with_multiple_tasks(self):
        """Test executing workflow with multiple tasks."""
        # Create graph
        graph = TaskGraph()

        @task
        def task_a():
            return "A_result"

        @task
        def task_b():
            return "B_result"

        # Add tasks to graph
        graph.add_node(task_a, "task_a")
        graph.add_node(task_b, "task_b")
        graph.add_edge("task_a", "task_b")

        # Create context
        context = ExecutionContext.create(graph)
        task_a.set_execution_context(context)
        task_b.set_execution_context(context)

        # Add first task to queue
        context.add_to_queue(task_a)

        # Execute
        engine = WorkflowEngine()
        engine.execute(context)

        # Verify results
        assert context.get_result("task_a") == "A_result"
        assert context.get_result("task_b") == "B_result"

    def test_execute_task_with_exception(self):
        """Test handling exceptions during task execution."""
        # Create graph
        graph = TaskGraph()

        @task
        def failing_task():
            raise ValueError("Task failed")

        # Add task to graph
        graph.add_node(failing_task, "failing_task")

        # Create context
        context = ExecutionContext.create(graph)
        failing_task.set_execution_context(context)

        # Add task to queue
        context.add_to_queue(failing_task)

        # Execute and expect exception
        engine = WorkflowEngine()
        with pytest.raises(exceptions.GraflowRuntimeError) as exc_info:
            engine.execute(context)

        # Verify the wrapped exception contains the original ValueError
        assert exc_info.value.cause is not None
        assert isinstance(exc_info.value.cause, ValueError)
        assert str(exc_info.value.cause) == "Task failed"

        # Verify exception was stored
        result = context.get_result("failing_task")
        assert isinstance(result, ValueError)
        assert str(result) == "Task failed"
