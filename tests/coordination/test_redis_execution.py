"""Integration tests for Redis distributed execution."""

import json
import time
import unittest
from typing import List
from unittest.mock import MagicMock, patch

from graflow.coordination.records import SerializedTaskRecord
from graflow.coordination.redis import RedisCoordinator
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import Executable, TaskWrapper
from graflow.queue.redis import DistributedTaskQueue


class TestRedisExecution(unittest.TestCase):
    def setUp(self):
        self.redis_mock = MagicMock()
        # Mock pipeline to return self (chainable)
        self.redis_mock.pipeline.return_value = self.redis_mock

        # Mock pubsub
        self.pubsub_mock = MagicMock()
        self.redis_mock.pubsub.return_value = self.pubsub_mock

        self.graph = TaskGraph()
        # Use TaskWrapper for tasks with logic
        self.task1 = TaskWrapper("task1", lambda: "result1", register_to_context=False)
        self.task2 = TaskWrapper("task2", lambda: "result2", register_to_context=False)
        self.graph.add_node(self.task1)
        self.graph.add_node(self.task2)

        self.context = ExecutionContext(self.graph)

        # Set execution context on tasks (required for dispatch_task)
        self.task1.set_execution_context(self.context)
        self.task2.set_execution_context(self.context)

        # Setup RedisTaskQueue with mock
        self.queue = DistributedTaskQueue(
            self.context,
            redis_client=self.redis_mock,
            key_prefix="test"
        )

        self.coordinator = RedisCoordinator(self.queue)

    def test_execute_group_flow(self):
        """Test the full flow of execute_group."""
        group_id = "group1"
        tasks: List[Executable] = [self.task1, self.task2]
        handler = MagicMock()

        # Mock wait_barrier to return True immediately (or after some logic)
        # We need to simulate the worker side completing tasks

        # Mock redis.incr to simulate barrier count
        # First call: task1 completes -> 1
        # Second call: task2 completes -> 2
        # Third call: wait_barrier check -> 3 (if we call it)
        # But wait_barrier calls incr.

        # Let's mock wait_barrier to avoid complex redis mocking
        with patch.object(self.coordinator, 'wait_barrier', return_value=True) as _wait_mock:
            # Mock _get_completion_results
            with patch.object(self.coordinator, '_get_completion_results') as results_mock:
                results_mock.return_value = [
                    {"task_id": "task1", "success": True},
                    {"task_id": "task2", "success": True}
                ]

                self.coordinator.execute_group(group_id, tasks, self.context, handler)

                # Verify Graph Saved
                self.assertIsNotNone(self.context.graph_hash)
                # Check if graph was saved to Redis (compressed)
                self.redis_mock.set.assert_called()

                # Verify Dispatch
                # Should have dispatched 2 tasks
                self.assertEqual(self.redis_mock.lpush.call_count, 2)

                # Check content of dispatched records
                calls = self.redis_mock.lpush.call_args_list
                for call in calls:
                    args, _ = call
                    key, value = args
                    self.assertEqual(key, "test:queue")
                    record_data = json.loads(value)
                    self.assertEqual(record_data['graph_hash'], self.context.graph_hash)
                    self.assertIn(record_data['task_id'], ["task1", "task2"])
                    self.assertEqual(record_data['group_id'], group_id)

                # Verify Barrier Created
                self.redis_mock.set.assert_any_call(
                    f"{self.queue.key_prefix}:barrier:{group_id}:expected", 2
                )

                # Verify Handler Called
                handler.on_group_finished.assert_called_once()

    def test_worker_dequeue_record(self):
        """Test worker dequeuing a SerializedTaskRecord."""
        # Setup a record
        record = SerializedTaskRecord(
            task_id="task1",
            session_id="sess1",
            graph_hash="hash1",
            trace_id="trace1",
            created_at=time.time()
        )

        # Mock Redis lpop to return the record
        self.redis_mock.lpop.return_value = record.to_json()

        # Mock GraphStore load
        self.queue.graph_store = MagicMock()
        self.queue.graph_store.load.return_value = self.graph
        self.coordinator.graph_store = self.queue.graph_store

        # Dequeue
        task_spec = self.queue.dequeue()
        assert task_spec is not None

        # Verify
        self.assertIsNotNone(task_spec)
        self.assertEqual(task_spec.task_id, "task1")
        self.assertEqual(task_spec.execution_context.graph_hash, "hash1")
        self.assertEqual(task_spec.execution_context.session_id, "sess1")

        # Verify Graph Loaded
        self.coordinator.graph_store.load.assert_called_with("hash1")

if __name__ == '__main__':
    unittest.main()
