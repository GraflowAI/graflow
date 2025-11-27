"""Unit tests for GraphStore."""

import hashlib
import unittest
import zlib
from unittest.mock import MagicMock

from redis import Redis

from graflow.coordination.graph_store import GraphStore
from graflow.core.graph import TaskGraph
from graflow.core.serialization import dumps


class TestGraphStore(unittest.TestCase):
    def setUp(self):
        self.redis_mock = MagicMock(spec=Redis)
        self.graph_store = GraphStore(self.redis_mock, "test_prefix")
        self.sample_graph = TaskGraph()

    def test_save_graph(self):
        # Calculate expected hash
        graph_bytes = dumps(self.sample_graph)
        expected_hash = hashlib.sha256(graph_bytes).hexdigest()
        expected_compressed = zlib.compress(graph_bytes, level=6)

        # Execute save
        graph_hash = self.graph_store.save(self.sample_graph)

        # Verify hash
        self.assertEqual(graph_hash, expected_hash)

        # Verify Redis interaction
        self.redis_mock.set.assert_called_once()
        args, kwargs = self.redis_mock.set.call_args
        self.assertEqual(args[0], f"test_prefix:graph:{expected_hash}")
        self.assertEqual(args[1], expected_compressed)
        self.assertIs(kwargs['nx'], True)
        self.assertEqual(kwargs['ex'], GraphStore.DEFAULT_TTL)

        # Verify local cache
        self.assertIn(graph_hash, self.graph_store._local_cache)
        self.assertEqual(self.graph_store._local_cache[graph_hash], self.sample_graph)

    def test_save_graph_cached(self):
        # First save
        graph_hash = self.graph_store.save(self.sample_graph)
        self.redis_mock.set.reset_mock()

        # Second save (should hit cache)
        graph_hash_2 = self.graph_store.save(self.sample_graph)

        self.assertEqual(graph_hash, graph_hash_2)
        self.redis_mock.set.assert_not_called()

    def test_load_graph(self):
        # Prepare mock data
        graph_bytes = dumps(self.sample_graph)
        graph_hash = hashlib.sha256(graph_bytes).hexdigest()
        compressed = zlib.compress(graph_bytes, level=6)

        self.redis_mock.getex.return_value = compressed

        # Execute load
        loaded_graph = self.graph_store.load(graph_hash)

        # Verify loaded graph structure (simple check)
        self.assertIsInstance(loaded_graph, TaskGraph)

        # Verify Redis interaction
        self.redis_mock.getex.assert_called_once_with(
            f"test_prefix:graph:{graph_hash}", ex=GraphStore.DEFAULT_TTL
        )

        # Verify local cache update
        self.assertIn(graph_hash, self.graph_store._local_cache)

    def test_load_graph_cached(self):
        # Pre-populate cache
        graph_bytes = dumps(self.sample_graph)
        graph_hash = hashlib.sha256(graph_bytes).hexdigest()
        self.graph_store._local_cache[graph_hash] = self.sample_graph

        # Execute load
        loaded_graph = self.graph_store.load(graph_hash)

        self.assertEqual(loaded_graph, self.sample_graph)
        self.redis_mock.get.assert_not_called()

    def test_load_graph_not_found(self):
        self.redis_mock.getex.return_value = None
        graph_hash = "non_existent_hash"

        with self.assertRaises(ValueError) as cm:
            self.graph_store.load(graph_hash)

        self.assertIn("Graph snapshot not found", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
