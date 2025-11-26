"""Tests for Redis TaskQueue implementation."""

import json
import time
from unittest.mock import Mock, patch

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import Executable
from graflow.queue.base import TaskSpec, TaskStatus
from graflow.queue.redis import DistributedTaskQueue


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = Mock()
    redis_mock.rpush.return_value = 1
    redis_mock.set.return_value = True
    redis_mock.lpop.return_value = None
    redis_mock.llen.return_value = 0
    redis_mock.lindex.return_value = None
    redis_mock.delete.return_value = 2
    return redis_mock


@pytest.fixture
def execution_context():
    """Create ExecutionContext for testing."""
    graph = TaskGraph()
    return ExecutionContext(graph)


class DummyExecutable(Executable):
    """Serializable executable used for TaskSpec tests."""

    def __init__(self, task_id: str):
        super().__init__()
        self._task_id = task_id
        self.__name__ = f"dummy_{task_id}"
        self.__qualname__ = self.__name__
        self.__module__ = __name__

    @property
    def task_id(self) -> str:
        return self._task_id

    def run(self):
        return f"result:{self._task_id}"

    def __call__(self):
        return self.run()


def create_registered_task(execution_context: ExecutionContext, task_id: str) -> DummyExecutable:
    """Create executable and add to graph.

    Tasks are stored in Graph, no separate registration needed.
    """
    task = DummyExecutable(task_id)
    # Add task to workflow graph for engine execution
    execution_context.graph.add_node(task, task_id)
    return task


class TestRedisTaskQueue:
    """Test Redis TaskQueue implementation."""

    def test_redis_not_available(self, execution_context):
        """Test graceful handling when redis is not available."""
        with patch('graflow.queue.redis.redis', None):
            with pytest.raises(ImportError, match="Redis library not installed"):
                from graflow.queue.redis import DistributedTaskQueue
                DistributedTaskQueue()

    @patch('graflow.queue.redis.redis')
    def test_redis_taskqueue_creation_default_client(self, mock_redis_module, execution_context):
        """Test RedisTaskQueue creation with default client."""
        mock_redis_client = Mock()
        mock_redis_module.Redis.return_value = mock_redis_client

        queue = DistributedTaskQueue()

        # execution_context no longer stored in queue
        assert queue.redis_client == mock_redis_client
        assert queue.key_prefix == "graflow"
        assert queue.queue_key == "graflow:queue"
        assert queue.specs_key == "graflow:specs"

        mock_redis_module.Redis.assert_called_once_with(
            host='localhost', port=6379, db=0, decode_responses=True
        )

    def test_redis_taskqueue_creation_custom_client(self, mock_redis, execution_context):
        """Test RedisTaskQueue creation with custom client."""
        with patch('graflow.queue.redis.redis'):
            queue = DistributedTaskQueue(redis_client=mock_redis, key_prefix="test_prefix")

            assert queue.redis_client == mock_redis
            assert queue.key_prefix == "test_prefix"
            assert queue.queue_key == "test_prefix:queue"
            assert queue.specs_key == "test_prefix:specs"

    def test_enqueue(self, mock_redis, execution_context):
        """Test enqueuing TaskSpec to Redis."""
        with patch('graflow.queue.redis.redis'):
            queue = DistributedTaskQueue(mock_redis)

            task = create_registered_task(execution_context, "test_node")
            execution_context.session_id = "sess-1"
            execution_context.trace_id = "trace-1"
            execution_context.graph_hash = "graph-xyz"
            task_spec = TaskSpec(
                executable=task,
                execution_context=execution_context,
                status=TaskStatus.READY,
                created_at=1234567890.0
            )

            result = queue.enqueue(task_spec)

            assert result is True
            assert queue._task_specs["test_node"] == task_spec

            # Verify Redis calls and serialized payload
            mock_redis.rpush.assert_called_once()
            queue_key, payload = mock_redis.rpush.call_args[0]
            assert queue_key == queue.queue_key

            record = json.loads(payload)
            assert record["task_id"] == "test_node"
            assert record["session_id"] == "sess-1"
            assert record["graph_hash"] == "graph-xyz"
            assert record["trace_id"] == "trace-1"
            assert record["group_id"] is None
            assert record["parent_span_id"] is None
            assert record["created_at"] == 1234567890.0

    def test_dequeue_empty_queue(self, mock_redis, execution_context):
        """Test dequeue from empty queue."""
        mock_redis.lpop.return_value = None

        with patch('graflow.queue.redis.redis'):
            queue = DistributedTaskQueue(mock_redis)

            result = queue.dequeue()

            assert result is None
            mock_redis.lpop.assert_called_once_with(queue.queue_key)

    def test_dequeue_success(self, mock_redis, execution_context):
        """Test successful dequeue operation."""
        record = {
            "task_id": "test_node",
            "session_id": "sess-1",
            "graph_hash": "graph-abc",
            "trace_id": "trace-1",
            "group_id": None,
            "parent_span_id": "parent-1",
            "created_at": 1234567890.0,
        }
        mock_redis.lpop.return_value = json.dumps(record)

        with patch('graflow.queue.redis.redis'):
            queue = DistributedTaskQueue(mock_redis)
            executable = create_registered_task(execution_context, "test_node")

            with patch(
                "graflow.worker.context_factory.ExecutionContextFactory.create_from_record",
                return_value=(execution_context, executable),
            ) as mock_factory:
                result = queue.dequeue()

            assert result is not None
            assert result.task_id == "test_node"
            assert result.status == TaskStatus.RUNNING
            assert result.created_at == 1234567890.0
            assert result.execution_context == execution_context
            assert result.trace_id == "trace-1"
            assert result.parent_span_id == "parent-1"
            assert queue._task_specs["test_node"] == result

            mock_factory.assert_called_once()
            args, kwargs = mock_factory.call_args
            assert args[0].task_id == "test_node"
            assert args[0].graph_hash == "graph-abc"
            assert mock_factory.call_args[0][1] is queue.graph_store
            mock_redis.lpop.assert_called_once_with(queue.queue_key)

    def test_is_empty(self, mock_redis, execution_context):
        """Test is_empty method."""
        with patch('graflow.queue.redis.redis'):
            queue = DistributedTaskQueue(mock_redis)

            # Test empty queue
            mock_redis.llen.return_value = 0
            assert queue.is_empty() is True

            # Test non-empty queue
            mock_redis.llen.return_value = 3
            assert queue.is_empty() is False

            # Verify Redis calls
            assert mock_redis.llen.call_count == 2
            mock_redis.llen.assert_called_with(queue.queue_key)

    def test_size(self, mock_redis, execution_context):
        """Test size method."""
        mock_redis.llen.return_value = 5

        with patch('graflow.queue.redis.redis'):
            queue = DistributedTaskQueue(mock_redis)

            result = queue.size()

            assert result == 5
            mock_redis.llen.assert_called_once_with(queue.queue_key)

    def test_peek_next_node(self, mock_redis, execution_context):
        """Test peek_next_node method."""
        with patch('graflow.queue.redis.redis'):
            queue = DistributedTaskQueue(mock_redis)

            # Test empty queue
            mock_redis.lindex.return_value = None
            result = queue.peek_next_node()
            assert result is None

            # Test queue with items
            mock_redis.lindex.return_value = json.dumps({"task_id": "test_node"})
            result = queue.peek_next_node()
            assert result == "test_node"

            # Verify Redis calls
            assert mock_redis.lindex.call_count == 2
            mock_redis.lindex.assert_called_with(queue.queue_key, 0)

    def test_cleanup(self, mock_redis, execution_context):
        """Test cleanup method."""
        with patch('graflow.queue.redis.redis'):
            queue = DistributedTaskQueue(mock_redis)

            queue.cleanup()

            mock_redis.delete.assert_called_once_with(queue.queue_key, queue.specs_key)


class TestRedisTaskQueueIntegration:
    """Integration tests for Redis TaskQueue with ExecutionContext."""

    def test_execution_context_with_separate_redis_queue(self, mock_redis):
        """Ensure ExecutionContext uses the in-memory queue while allowing a separate RedisTaskQueue."""
        graph = TaskGraph()

        with patch('graflow.queue.redis.redis'):
            # Add start node to graph first
            task_start = create_registered_task(ExecutionContext(graph), "start")

            context = ExecutionContext(
                graph,
                start_node="start",
                config={
                    'redis_client': mock_redis,
                    'key_prefix': 'test'
                }
            )

            assert context.task_queue.__class__.__name__ == "InMemoryTaskQueue"

            queue = DistributedTaskQueue(redis_client=mock_redis, key_prefix="test")

            assert isinstance(queue, DistributedTaskQueue)
            assert queue.redis_client == mock_redis
            assert queue.key_prefix == 'test'

    def test_redis_backend_compatibility_methods(self, mock_redis):
        """Test that Redis backend maintains compatibility with ExecutionContext."""
        graph = TaskGraph()

        with patch('graflow.queue.redis.redis'):
            # Add start node to graph first
            task_start = create_registered_task(ExecutionContext(graph), "start")

            context = ExecutionContext(
                graph,
                start_node="start",
                config={'redis_client': mock_redis}
            )

            queue = DistributedTaskQueue(redis_client=mock_redis)
            context.graph_hash = "graph-123"

            task1 = create_registered_task(context, "task1")

            queue.enqueue(TaskSpec(executable=task_start, execution_context=context))
            queue.enqueue(TaskSpec(executable=task1, execution_context=context))

            mock_redis.lpop.side_effect = [
                json.dumps(
                    {
                        "task_id": "start",
                        "session_id": context.session_id,
                        "graph_hash": "graph-123",
                        "trace_id": context.trace_id,
                        "group_id": None,
                        "parent_span_id": None,
                        "created_at": time.time(),
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task1",
                        "session_id": context.session_id,
                        "graph_hash": "graph-123",
                        "trace_id": context.trace_id,
                        "group_id": None,
                        "parent_span_id": None,
                        "created_at": time.time(),
                    }
                ),
                None,
            ]

            with patch(
                "graflow.worker.context_factory.ExecutionContextFactory.create_from_record",
                side_effect=[
                    (context, task_start),
                    (context, task1),
                ],
            ):
                node1 = queue.get_next_task()
                node2 = queue.get_next_task()
                node3 = queue.get_next_task()

            assert node1 == "start"
            assert node2 == "task1"
            assert node3 is None
