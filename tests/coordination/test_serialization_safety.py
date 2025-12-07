"""Test serialization safety of GraphStore."""

from unittest.mock import MagicMock

import pytest

from graflow.coordination.graph_store import GraphStore
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper


class TestGraphSerialization:
    """Test graph serialization with execution context references."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.redis_mock = MagicMock()
        self.graph_store = GraphStore(self.redis_mock, "test")
        self.graph = TaskGraph()

        # Create a task with logic
        self.task = TaskWrapper("task1", lambda: "result", register_to_context=False)
        self.graph.add_node(self.task)

        # Create ExecutionContext and set it on task
        self.context = ExecutionContext(self.graph)
        self.task.set_execution_context(self.context)

    def test_save_graph_with_context(self):
        """Test saving a graph where tasks have references to ExecutionContext.

        Verifies that Executable.__getstate__() properly strips execution context
        so that tasks can be serialized even when they have context references.
        """
        # Verify task has execution context set
        assert hasattr(self.task, '_execution_context')
        assert self.task.get_execution_context() is self.context

        # Save should succeed because __getstate__ strips _execution_context
        graph_hash = self.graph_store.save(self.graph)
        assert graph_hash is not None

        # Verify original task still has context (save shouldn't mutate it)
        assert hasattr(self.task, '_execution_context')
        assert self.task.get_execution_context() is self.context
