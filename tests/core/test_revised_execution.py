"""Tests for revised execution functionality."""

from graflow.core.context import execute_with_cycles
from graflow.core.decorators import task
from graflow.core.task import Task
from graflow.core.workflow import clear_workflow_context, current_workflow_context, workflow


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

    context = current_workflow_context()  # Ensure context is initialized

    graph = context.graph
    assert "start" in graph.nodes, f"start not in graph nodes: {list(graph.nodes)}"
    assert "end" in graph.nodes, f"end not in graph nodes: {list(graph.nodes)}"

    start >> end # type: ignore

    # Execute the flow
    execute_with_cycles(graph, "start", max_steps=5)

    assert "start" in executed
    assert "end" in executed


def test_execute_with_traditional_tasks():
    """Test execution with traditional Task objects."""
    clear_workflow_context()
    context = current_workflow_context()

    A = Task("A")  # noqa: N806
    B = Task("B")  # noqa: N806
    C = Task("C")  # noqa: N806

    A >> B >> C # type: ignore

    # Should execute without error
    execute_with_cycles(context.graph, "A", max_steps=5)


def test_parallel_group_execution():
    """Test execution with parallel groups."""
    clear_workflow_context()
    context = current_workflow_context()

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
    start >> (parallel1 | parallel2) >> end # type: ignore

    execute_with_cycles(context.graph, "start", max_steps=10)

    assert "start" in executed
    # Both parallel tasks should be executed within the group


def test_global_graph_state():
    """Test that global graph maintains state correctly."""
    clear_workflow_context()

    with workflow("test_global_graph_state") as context:
        @task
        def task1():
            pass

        @task
        def task2():
            pass

        task1 >> task2 # type: ignore

        graph = context.graph.nx_graph()
        assert "task1" in graph.nodes
        assert "task2" in graph.nodes
        assert ("task1", "task2") in graph.edges()


def test_clear_graph():
    """Test clearing the global graph."""
    clear_workflow_context()
    context = current_workflow_context()

    @task
    def temp_task():
        pass

    graph = context.graph
    assert len(graph.nodes) > 0

    clear_workflow_context()

    context = current_workflow_context()
    graph = context.graph
    assert len(graph.nodes) == 0
