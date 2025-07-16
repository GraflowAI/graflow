"""Graph construction functionality for graflow."""

from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx

from graflow.core.task import Executable
from graflow.core.workflow import WorkflowContext, get_current_workflow_context


def build_graph(start_node: Executable, context: Optional[WorkflowContext] = None) -> nx.DiGraph:
    """Build a NetworkX directed graph from an executable."""
    if context is None:
        # Use the current workflow context if not provided
        context = get_current_workflow_context()

    graph = context.graph
    new_graph: nx.DiGraph = nx.DiGraph()
    visited: set[str] = set()

    def _build_graph_recursive(node: Executable) -> None:
        """Recursively build the graph from the executable."""
        if node.name in visited:
            return
        visited.add(node.name)

        new_graph.add_node(node.name, task=node)

        for successor in graph.successors(node.name):
            successor_task = graph.nodes[successor]["task"]
            new_graph.add_edge(node.name, successor)
            _build_graph_recursive(successor_task)

        for predecessor in graph.predecessors(node.name):
            predecessor_task = graph.nodes[predecessor]["task"]
            new_graph.add_edge(predecessor, node.name)
            _build_graph_recursive(predecessor_task)

    _build_graph_recursive(start_node)
    return new_graph


def draw_task_graph(graph: nx.DiGraph, title: str = "Task Graph") -> None:
    """Draw a task graph using matplotlib."""
    pos = nx.spring_layout(graph)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=2000,
        edge_color="black",
        arrows=True,
    )
    plt.title(title)
    plt.show()


def visualize_dependencies(graph: nx.DiGraph) -> None:
    """Visualize task dependencies."""
    print("=== Dependencies ===")
    for node in graph.nodes():
        successors = list(graph.successors(node))
        if successors:
            print(f"{node} >> {' >> '.join(successors)}")
        else:
            print(f"{node} (no dependencies)")


def show_graph_info(graph: nx.DiGraph) -> None:
    """Display information about the task graph."""

    print("=== Graph Information ===")
    print(f"Nodes: {list(graph.nodes())}")
    print(f"Edges: {list(graph.edges())}")

    # Cycle detection
    try:
        cycles = list(nx.simple_cycles(graph))
        if cycles:
            print(f"Cycles detected: {cycles}")
        else:
            print("No cycles detected")
    except Exception:
        print("Error detecting cycles")
