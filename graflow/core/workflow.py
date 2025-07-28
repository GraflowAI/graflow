"""Workflow context manager for graflow."""

from __future__ import annotations

import contextvars
import uuid
from typing import TYPE_CHECKING, Optional

from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.exceptions import GraphCompilationError

if TYPE_CHECKING:
    from .task import Executable

# Context variable for current workflow context
_current_context: contextvars.ContextVar[Optional[WorkflowContext]] = contextvars.ContextVar(
    'current_workflow', default=None
)

class WorkflowContext:
    """
    Context for Workflow definition and scoped task registration.
    This class manages the workflow graph and provides methods to add tasks,
    edges, and execute the workflow."""

    def __init__(self, name: str):
        """Initialize a new workflow context.

        Args:
            name: Name for this workflow
        """
        self.name = name
        self.graph = TaskGraph()
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
        self.graph.add_node(name, task)

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add an edge between tasks in this workflow's graph."""
        self.graph.add_edge(from_node, to_node)

    def execute(self, start_node: Optional[str] = None, max_steps: int = 10) -> None:
        """Execute the workflow starting from the specified node."""
        if start_node is None:
            # Find start nodes (nodes with no predecessors)
            candidate_nodes = self.graph.get_start_nodes()
            if not candidate_nodes:
                raise GraphCompilationError("No start node specified and no nodes with no predecessors found.")
            elif len(candidate_nodes) > 1:
                raise GraphCompilationError("Multiple start nodes found, please specify one.")
            start_node = candidate_nodes[0]
            assert start_node is not None, "No valid start node found"

        from .engine import WorkflowEngine  # noqa: PLC0415

        exec_context = ExecutionContext.create(self.graph, start_node, max_steps=max_steps)
        engine = WorkflowEngine()
        engine.execute(exec_context)

    def show_info(self) -> None:
        """Display information about this workflow's graph."""
        print(f"=== Workflow '{self.name}' Information ===")
        print(f"Nodes: {self.graph.nodes}")
        print(f"Edges: {self.graph.get_edges()}")

        # Cycle detection
        cycles = self.graph.detect_cycles()
        if cycles:
            print(f"Cycles detected: {cycles}")
        else:
            print("No cycles detected")

    def visualize_dependencies(self) -> None:
        """Visualize task dependencies in this workflow."""
        print(f"=== Workflow '{self.name}' Dependencies ===")
        for node in self.graph.nodes:
            successors = self.graph.successors(node)
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

def get_current_workflow_context() -> WorkflowContext:
    """Get the current workflow context if any."""
    ctx = _current_context.get()
    if ctx is None:
        name = uuid.uuid4().hex
        ctx = WorkflowContext(name)
        _current_context.set(ctx)
    return ctx

def set_current_workflow_context(context: WorkflowContext) -> None:
    """Set the current workflow context."""
    _current_context.set(context)

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
