"""Task graph management for graflow workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import networkx as nx
from networkx.classes.reportviews import EdgeView, NodeView

from graflow.exceptions import DuplicateTaskError

if TYPE_CHECKING:
    from .task import Executable


class TaskGraph:
    """Manages the task graph for a workflow."""

    def __init__(self):
        """Initialize a new task graph."""
        self._graph: nx.DiGraph = nx.DiGraph()

    def nx_graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph

    def add_task(self, task: Executable) -> None:
        """Add a task node to the graph."""
        self.add_node(task.task_id, task)

    def add_node(self, task_id: str, task: Executable) -> None:
        """Add a task node to the graph."""
        if task_id in self._graph.nodes:
            raise DuplicateTaskError(task_id)
        self._graph.add_node(task_id, task=task)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between tasks in the graph."""
        self._graph.add_edge(from_node, to_node)

    def get_node(self, task_id: str) -> Executable:
        """Get a task node by its ID."""
        if task_id not in self._graph.nodes:
            raise KeyError(f"Task {task_id} not found in graph")
        return self._graph.nodes[task_id]["task"]

    @property
    def nodes(self) -> NodeView:
        """Get all node names in the graph."""
        return self._graph.nodes()

    @property
    def edges(self) -> EdgeView:
        """Get all edges in the graph."""
        return self._graph.edges()

    def get_edges(self) -> List[tuple]:
        """Get all edges in the graph."""
        return list(self._graph.edges())

    def get_start_nodes(self) -> List[str]:
        """Get nodes with no predecessors (start nodes)."""
        return [node for node in self._graph.nodes() if self._graph.in_degree(node) == 0]

    def successors(self, node: str) -> List[str]:
        """Get successor nodes of the given node."""
        return list(self._graph.successors(node))

    def predecessors(self, node: str) -> List[str]:
        """Get predecessor nodes of the given node."""
        return list(self._graph.predecessors(node))

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the graph."""
        try:
            return list(nx.simple_cycles(self._graph))
        except Exception:
            return []

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self._graph.clear()

    def __str__(self) -> str:
        """Return a string representation of the graph."""
        from graflow.utils.graph import draw_ascii
        return draw_ascii(self._graph)
