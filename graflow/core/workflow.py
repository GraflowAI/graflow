"""Workflow context manager for graflow."""

from __future__ import annotations

import contextvars
import uuid
from typing import TYPE_CHECKING, Optional

import networkx as nx

from .context import ExecutionContext

if TYPE_CHECKING:
    from .task import Executable

# Context variable for current workflow context
_current_context: contextvars.ContextVar[Optional[WorkflowContext]] = contextvars.ContextVar(
    'current_workflow', default=None
)

class WorkflowContext:
    """Context manager for workflow execution with scoped task registration."""

    def __init__(self, name: str):
        """Initialize a new workflow context.

        Args:
            name: Name for this workflow
        """
        self.name = name
        self.graph: nx.DiGraph = nx.DiGraph()
        self._task_counter = 0
        self._group_counter = 0

    def __enter__(self):
        """Enter the workflow context."""
        # Store previous context if any
        self._previous_context = _current_context.get()
        # Set this as current context
        _current_context.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the workflow context."""
        # Restore previous context
        _current_context.set(self._previous_context)

    def add_node(self, name: str, task: Executable) -> None:
        """Add a task node to this workflow's graph."""
        self.graph.add_node(name, task=task)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between tasks in this workflow's graph."""
        self.graph.add_edge(from_node, to_node)

    def execute(self, start_node: str, max_steps: int = 10) -> None:
        """Execute the workflow starting from the specified node."""
        exec_context = ExecutionContext.create(self.graph, start_node, max_steps=max_steps)
        exec_context.execute()

    def show_info(self) -> None:
        """Display information about this workflow's graph."""
        print(f"=== Workflow '{self.name}' Information ===")
        print(f"Nodes: {list(self.graph.nodes())}")
        print(f"Edges: {list(self.graph.edges())}")

        # Cycle detection
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                print(f"Cycles detected: {cycles}")
            else:
                print("No cycles detected")
        except Exception:
            print("Error detecting cycles")

    def visualize_dependencies(self) -> None:
        """Visualize task dependencies in this workflow."""
        print(f"=== Workflow '{self.name}' Dependencies ===")
        for node in self.graph.nodes():
            successors = list(self.graph.successors(node))
            if successors:
                print(f"{node} >> {' >> '.join(successors)}")
            else:
                print(f"{node} (no dependencies)")

    def clear(self) -> None:
        """Clear all tasks from this workflow."""
        self.graph.clear()
        self._task_counter = 0
        self._group_counter = 0

    def get_next_group_name(self) -> str:
        """Get the next group name for this workflow."""
        self._group_counter += 1
        return f"ParallelGroup_{self._group_counter}"

    def task(self, func=None, *, name=None):
        """Context-scoped task decorator."""
        from .decorators import task as _task  # noqa: PLC0415
        return _task(func, name=name)


def get_current_workflow_context() -> WorkflowContext:
    """Get the current workflow context if any."""
    ctx = _current_context.get()
    if ctx is None:
        name = uuid.uuid4().hex
        ctx = WorkflowContext(name)
        _current_context.set(ctx)
    return ctx

def clear_workflow_context() -> None:
    """Clear the current workflow context."""
    _current_context.set(None)

def workflow(name: str) -> WorkflowContext:
    """Context manager for creating a workflow.

    Args:
        name: Name of the workflow

    Returns:
        WorkflowContext instance
    """
    return WorkflowContext(name)
