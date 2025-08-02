"""Tests for graph construction functionality."""

import networkx as nx
import pytest

from graflow.core.task import ParallelGroup, Task
from graflow.core.workflow import clear_workflow_context
from graflow.exceptions import DuplicateTaskError
from graflow.utils.graph import build_graph


@pytest.fixture(autouse=True)
def reset_workflow_context():
    """Reset workflow context before each test."""
    clear_workflow_context()


def test_single_task_graph():
    """Test graph construction for a single task."""
    task = Task("A")
    graph = build_graph(task)

    assert list(graph.nodes) == ["A"]
    assert list(graph.edges) == []
    assert graph.nx_graph().nodes["A"]["task"] == task


def test_graph_is_directed():
    """Test that constructed graph is directed."""
    task_a = Task("A")
    task_b = Task("B")
    flow = task_a >> task_b
    graph = build_graph(flow)

    assert isinstance(graph, nx.DiGraph)
    assert graph.is_directed()


def test_parallel_group_graph():
    """Test graph construction for parallel group."""
    task_a = Task("A")
    task_b = Task("B")
    group = ParallelGroup([task_a, task_b])
    graph = build_graph(group)

    # draw_task_graph(graph, title="Parallel Group Graph")

    assert set(graph.nodes) == {group.task_id, "A", "B"}
    expected_edges = {(group.task_id, "A"), (group.task_id, "B")}
    assert set(graph.edges) == expected_edges

def test_cycle_detection():
    """Test cycle detection in a graph."""
    task_a = Task("A")
    task_b = Task("B")
    task_c = Task("C")

    # Create a cycle: A >> B >> C >> A
    flow = task_a >> task_b >> task_c >> task_a
    graph = build_graph(flow)

    # draw_task_graph(graph, title="Parallel Group Graph")

    # Check for cycles
    cycles = list(nx.simple_cycles(graph.nx_graph()))
    assert len(cycles) == 1
    assert set(cycles[0]) == {"A", "B", "C"}

def test_a_a_b():
    """Test graph construction for flow: A >> A >> B."""
    task_a1 = Task("A")

    with pytest.raises(DuplicateTaskError):
        task_a2 = Task("A")
        assert task_a1.task_id == task_a2.task_id, "Duplicate task IDs should match"

def test_a_b_a():
    """Test graph construction for flow: A >> B >> A."""
    task_a1 = Task("A")
    task_b = Task("B")

    flow = task_a1 >> task_b >> task_a1
    graph = build_graph(flow)

    # draw_task_graph(graph, title="A_B_A Graph")

    assert set(graph.nodes) == {"A", "B"}
    expected_edges = {
        ("A", "B"),
        ("B", "A"),
    }
    assert set(graph.edges) == expected_edges

def test_complex_graph1():
    """Test graph construction for complex flow: A >> (B | C) >> D."""
    task_a = Task("A")
    task_b = Task("B")
    task_c = Task("C")
    task_d = Task("D")

    flow = task_a >> (task_b | task_c) >> task_d
    graph = build_graph(flow)

    #draw_task_graph(graph, title="Parallel Group Graph")

    assert set(graph.nodes) == {"A", "B", "C", "D", "ParallelGroup_1"}
    expected_edges = {
        ("A", "ParallelGroup_1"),
        ("ParallelGroup_1", "B"),
        ("ParallelGroup_1", "C"),
        ("B", "D"),
        ("C", "D")
    }
    assert set(graph.edges) == expected_edges

def test_merge_parallel_groups():
    """Test merging two parallel groups."""
    task_a = Task("A")
    task_b = Task("B")
    task_c = Task("C")

    group1 = ParallelGroup([task_a, task_b])
    group2 = ParallelGroup([task_b, task_c])
    merged_group = group1 | group2

    graph = build_graph(merged_group)

    # draw_task_graph(graph, title="Merged Parallel Groups Graph")

    expected_nodes = {"A", "B", "C", merged_group.task_id}
    assert set(graph.nodes) == expected_nodes
    expected_edges = {
        (merged_group.task_id, "A"),
        (merged_group.task_id, "B"),
        (merged_group.task_id, "C"),
    }
    assert set(graph.edges) == expected_edges


def test_merge_parallel_groups2():
    """Test merging two parallel groups."""
    task_a = Task("A")
    task_b = Task("B")
    task_c = Task("C")
    task_d = Task("D")
    task_e = Task("E")

    pipeline = (task_a | task_b) | (task_c | task_d | task_e)

    graph = build_graph(pipeline)

    # draw_task_graph(graph, title="Merged Parallel Groups Graph")

    expected_nodes = {"A", "B", "C", "D", "E", pipeline.task_id}
    assert set(graph.nodes) == expected_nodes
    expected_edges = {
        (pipeline.task_id, "A"),
        (pipeline.task_id, "B"),
        (pipeline.task_id, "C"),
        (pipeline.task_id, "D"),
        (pipeline.task_id, "E"),
    }
    assert set(graph.edges) == expected_edges


def test_complex_graph2():
    """Test graph construction for complex flow: A >> (B | C) >> (D | E | F)."""
    task_a = Task("A")
    task_b = Task("B")
    task_c = Task("C")
    task_d = Task("D")
    task_e = Task("E")
    task_f = Task("F")

    group1 = task_b | task_c
    group2 = task_d | task_e | task_f
    flow = task_a >> group1 >> group2
    graph = build_graph(flow)

    # draw_task_graph(graph, title="Complex Parallel Group Graph")

    expected_nodes = {"A", "B", "C", "D", "E", "F", group1.task_id, group2.task_id}
    assert set(graph.nodes) == expected_nodes
    expected_edges = {
        ("A", group1.task_id),
        (group1.task_id, "B"), (group1.task_id, "C"),
        ("B", group2.task_id), ("C", group2.task_id),
        (group2.task_id, "D"), (group2.task_id, "E"), (group2.task_id, "F"),
        }
    assert set(graph.edges) == expected_edges
