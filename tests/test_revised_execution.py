"""Tests for revised execution functionality."""

from graflow.core.context import execute_with_cycles
from graflow.core.decorators import task
from graflow.core.task import Task, clear_workflow_context


def test_execute_simple_flow():
    """Test execution of simple task flow."""
    executed = []

    clear_workflow_context()  # Clear first

    @task
    def start():
        executed.append("start")

    @task
    def end():
        executed.append("end")

    # Check graph state
    graph = get_task_graph()
    assert "start" in graph.nodes(), f"start not in graph nodes: {list(graph.nodes())}"
    assert "end" in graph.nodes(), f"end not in graph nodes: {list(graph.nodes())}"

    start >> end

    # Execute the flow
    execute_with_cycles("start", max_steps=5)

    assert "start" in executed
    assert "end" in executed


def test_execute_with_traditional_tasks():
    """Test execution with traditional Task objects."""
    clear_workflow_context()

    A = Task("A")
    B = Task("B")
    C = Task("C")

    A >> B >> C

    # Should execute without error
    execute_with_cycles("A", max_steps=5)


def test_parallel_group_execution():
    """Test execution with parallel groups."""
    clear_workflow_context()

    executed = []

    @task
    def start():
        executed.append("start")

    @task
    def parallel1():
        executed.append("parallel1")

    @task
    def parallel2():
        executed.append("parallel2")

    @task
    def end():
        executed.append("end")

    # Create flow with parallel group
    start >> (parallel1 | parallel2) >> end

    execute_with_cycles("start", max_steps=10)

    assert "start" in executed
    # Both parallel tasks should be executed within the group


def test_global_graph_state():
    """Test that global graph maintains state correctly."""
    clear_workflow_context()

    @task
    def task1():
        pass

    @task
    def task2():
        pass

    task1 >> task2

    graph = get_task_graph()
    assert "task1" in graph.nodes()
    assert "task2" in graph.nodes()
    assert ("task1", "task2") in graph.edges()


def test_clear_graph():
    """Test clearing the global graph."""
    clear_workflow_context()

    @task
    def temp_task():
        pass

    graph = get_task_graph()
    assert len(graph.nodes()) > 0

    clear_workflow_context()
    graph = get_task_graph()
    assert len(graph.nodes()) == 0
