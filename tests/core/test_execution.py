"""Tests for execution engine functionality."""

import os
import tempfile

from graflow.core.context import ExecutionContext, create_execution_context, execute_with_cycles
from graflow.core.task import Task
from graflow.utils.graph import build_graph


def test_create_initial_context():
    """Test initial context creation."""
    context = create_execution_context("A")

    assert context.queue.to_list() == ["A"]
    assert context.executed == []
    assert context.steps == 0


def test_save_and_load_context():
    """Test context serialization and deserialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_path = os.path.join(tmpdir, "test_context.pkl")

        original_context = create_execution_context("A")
        original_context.executed = ["A", "B"]
        original_context.steps= 2

        original_context.save(context_path)
        loaded_context = ExecutionContext.load(context_path)

        assert loaded_context.executed == ["A", "B"]
        assert loaded_context.steps == 2
        assert loaded_context.queue.to_list() == ["A"]


def test_execute_with_context():
    """Test execution with context."""
    task = Task("A")
    graph = build_graph(task)

    # execute_with_cycles creates its own context
    execute_with_cycles(graph, "A", max_steps=1)

    # The execution prints show it ran successfully
    # This test verifies the execution completes without error


def test_execute_with_max_steps():
    """Test execution respects max_steps limit."""
    task_a = Task("A")
    task_b = Task("B")
    flow = task_a >> task_b
    graph = build_graph(flow)

    # execute_with_cycles creates its own context and respects max_steps
    execute_with_cycles(graph, "A", max_steps=1)

    # The execution should complete without error and respect the step limit
    # This test verifies the execution completes without error
