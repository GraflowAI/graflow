"""Task graph management for graflow workflows."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

import networkx as nx
from networkx.classes.reportviews import NodeView, OutEdgeView

from graflow.exceptions import DuplicateTaskError

if TYPE_CHECKING:
    from .task import Executable

logger = logging.getLogger(__name__)


class TaskGraph:
    """Manages the task graph for a workflow."""

    def __init__(self):
        """Initialize a new task graph."""
        self._graph: nx.DiGraph = nx.DiGraph()

    def nx_graph(self) -> nx.DiGraph:
        """Get the underlying NetworkX graph."""
        return self._graph

    def add_node(self, task: Executable, task_id: Optional[str] = None, skip_if_exists: bool = False) -> None:
        """Add a task node to the graph.

        If a node with the same task_id already exists, logs a warning and skips adding.
        """
        if task_id is None:
            task_id = task.task_id
        if task_id in self._graph.nodes:
            if skip_if_exists:
                logger.debug("Node '%s' already exists in graph, skipping duplicate add_node", task_id)
                return
            else:
                raise DuplicateTaskError(f"Node '{task_id}' already exists in graph")
        self._graph.add_node(task_id, task=task)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between tasks in the graph."""
        self._graph.add_edge(from_node, to_node)

    def get_node(self, task_id: str) -> Executable:
        """Get a task node by its ID."""
        if task_id in self._graph.nodes:
            node_data = self._graph.nodes[task_id]
            if "task" not in node_data:
                raise KeyError(f"Node {task_id} exists but has no 'task' attribute")
            return node_data["task"]

        # Search in ParallelGroups
        from graflow.core.task import ParallelGroup

        for node_id in self._graph.nodes:
            node_data = self._graph.nodes[node_id]
            if "task" in node_data:
                task = node_data["task"]
                if isinstance(task, ParallelGroup):
                    for member in task.tasks:
                        if member.task_id == task_id:
                            return member

        raise KeyError(f"Task {task_id} not found in graph")

    @property
    def nodes(self) -> NodeView:
        """Get all node names in the graph."""
        return self._graph.nodes()  # type: ignore

    @property
    def edges(self) -> OutEdgeView:
        """Get all edges in the graph."""
        return self._graph.edges()

    def get_edges(self) -> List[tuple]:
        """Get all edges in the graph."""
        return list(self._graph.edges())

    def get_start_nodes(self) -> List[str]:
        """Get nodes with no predecessors (start nodes)."""
        start_nodes = [node for node in self._graph.nodes() if self._graph.in_degree(node) == 0]
        if len(start_nodes) <= 1:
            return start_nodes

        # Remove ParallelGroup members from start nodes
        from graflow.core.task import ParallelGroup

        for node_id, node_data in self._graph.nodes(data=True):
            if node_id in start_nodes:
                task = node_data.get("task")
                if isinstance(task, ParallelGroup):
                    for member in task.tasks:
                        start_nodes.remove(member.task_id)

        return start_nodes

    def successors(self, node: str) -> List[str]:
        """Get successor nodes of the given node."""
        return list(self._graph.successors(node))

    def predecessors(self, node: str) -> List[str]:
        """Get predecessor nodes of the given node."""
        return list(self._graph.predecessors(node))

    def get_parallel_group_members(self, group_id: str) -> List[str]:
        """Return task IDs for members of the given ParallelGroup."""
        node_data = self._graph.nodes.get(group_id)
        if not node_data:
            return []

        task = node_data.get("task")
        if task is None:
            return []

        from graflow.core.task import ParallelGroup  # Local import to avoid cycles

        if isinstance(task, ParallelGroup):
            return [member.task_id for member in task.tasks]

        return []

    def detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the graph."""
        try:
            return list(nx.simple_cycles(self._graph))
        except Exception:
            return []

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self._graph.clear()

    def rename_node(self, old_task_id: str, new_task_id: str) -> None:
        """Rename a node in the graph.

        Args:
            old_task_id: Current task ID to rename
            new_task_id: New task ID to assign

        Raises:
            KeyError: If old_task_id doesn't exist
            ValueError: If new_task_id already exists
        """
        if old_task_id not in self._graph.nodes:
            raise KeyError(f"Task {old_task_id} not found in graph")
        if new_task_id in self._graph.nodes:
            raise ValueError(f"Task {new_task_id} already exists in graph")

        # Use NetworkX relabel_nodes to rename the node
        mapping = {old_task_id: new_task_id}
        nx.relabel_nodes(self._graph, mapping, copy=False)

    def __str__(self) -> str:
        """Return a string representation of the graph."""
        from graflow.utils.graph import draw_ascii

        return draw_ascii(self._graph)
