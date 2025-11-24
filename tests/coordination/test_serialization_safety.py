"""Test serialization safety of GraphStore."""

import unittest
from unittest.mock import MagicMock

from graflow.coordination.graph_store import GraphStore
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper


class TestGraphSerialization(unittest.TestCase):
    def setUp(self):
        self.redis_mock = MagicMock()
        self.graph_store = GraphStore(self.redis_mock, "test")
        self.graph = TaskGraph()

        # Create a task with logic
        self.task = TaskWrapper("task1", lambda: "result", register_to_context=False)
        self.graph.add_node(self.task)

        # Create ExecutionContext with a REAL Redis client (unpicklable)
        # We use a real client but don't connect to avoid errors if redis is missing
        # Actually, MagicMock is picklable? No, usually not.
        # But let's use a real object that is definitely not picklable or problematic.
        # A file handle or a socket.

        # Or better, use a class that raises error on pickle
        class Unpicklable:
            def __getstate__(self):
                raise TypeError("Cannot pickle me")

        self.context = ExecutionContext(self.graph)
        # Inject unpicklable object into context config
        self.context._original_config = {"unpicklable": Unpicklable()}

        # Set context on task
        self.task.set_execution_context(self.context)

    def test_save_graph_with_context(self):
        """Test saving a graph where tasks have references to ExecutionContext."""
        # This should fail if we don't strip the context
        try:
            self.graph_store.save(self.graph)
        except TypeError as e:
            print(f"Caught expected TypeError: {e}")
            # If it fails, we know we need to fix it.
            # But we want to assert that it DOES NOT fail if we fix it.
            # So for now, let's see if it fails.
            self.fail("GraphStore.save failed due to unpicklable ExecutionContext")
        except Exception as e:
            self.fail(f"GraphStore.save failed with unexpected error: {e}")

if __name__ == '__main__':
    unittest.main()
