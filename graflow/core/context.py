"""Execution engine for graflow with cycle support and global graph integration."""

from __future__ import annotations

import pickle
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

from .engine import WorkflowEngine


@dataclass
class ExecutionContext:
    """Encapsulates execution state and provides execution methods."""

    # Execution state
    queue: deque = field(default_factory=deque)
    executed: List[str] = field(default_factory=list)
    steps: int = 0
    max_steps: int = 10
    start_node: Optional[str] = None

    # Execution results and data
    results: Dict[str, Any] = field(default_factory=dict)
    shared_data: Dict[str, Any] = field(default_factory=dict)

    # Graph reference
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def __post_init__(self):
        """Initialize context after creation."""
        if self.start_node and not self.queue:
            self.queue.append(self.start_node)

    @classmethod
    def create(cls, graph: nx.DiGraph, start_node: str, max_steps: int = 10) -> ExecutionContext:
        """Create a new execution context."""
        return cls(
            queue=deque([start_node]),
            start_node=start_node,
            max_steps=max_steps,
            graph=graph
        )

    def add_to_queue(self, node: str) -> None:
        """Add a node to the execution queue."""
        self.queue.append(node)

    def mark_executed(self, node: str) -> None:
        """Mark a node as executed."""
        if node not in self.executed:
            self.executed.append(node)

    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return not self.queue or self.steps >= self.max_steps

    def get_next_node(self) -> Optional[str]:
        """Get the next node to execute."""
        return self.queue.popleft() if self.queue else None

    def increment_step(self) -> None:
        """Increment the step counter."""
        self.steps += 1

    def set_result(self, node: str, result: Any) -> None:
        """Store execution result for a node."""
        self.results[node] = result

    def get_result(self, node: str) -> Any:
        """Get execution result for a node."""
        return self.results.get(node)

    def set_shared_data(self, key: str, value: Any) -> None:
        """Set shared data accessible to all tasks."""
        self.shared_data[key] = value

    def get_shared_data(self, key: str, default: Any = None) -> Any:
        """Get shared data."""
        return self.shared_data.get(key, default)

    def save(self, path: str = "execution_context.pkl") -> None:
        """Save execution context to a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "execution_context.pkl") -> ExecutionContext:
        """Load execution context from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def execute(self) -> None:
        """Execute tasks using this context."""
        engine = WorkflowEngine()
        engine.execute(self)

    def save_context(self, path: str = "context.pkl") -> None:
        """Save the current execution context to a file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

def create_execution_context(start_node: str = "ROOT", max_steps: int = 10) -> ExecutionContext:
    """Create an initial execution context with a single root node."""
    return ExecutionContext.create(nx.DiGraph(), start_node, max_steps=max_steps)

def load_context(path: str = "context.pkl") -> ExecutionContext:
    """Load execution context from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def execute_with_cycles(graph: nx.DiGraph, start_node: str, max_steps: int = 10) -> None:
    """Execute tasks allowing cycles from global graph."""
    engine = WorkflowEngine()
    engine.execute_with_cycles(graph, start_node, max_steps)

