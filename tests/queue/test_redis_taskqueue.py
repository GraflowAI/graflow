"""Tests for Redis TaskQueue implementation."""

import json
import time
from unittest.mock import Mock, patch

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.queue.base import TaskSpec, TaskStatus
from graflow.queue.factory import QueueBackend
from graflow.queue.redis import RedisTaskQueue


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = Mock()
    redis_mock.hset.return_value = True
    redis_mock.rpush.return_value = 1
    redis_mock.lpop.return_value = None
    redis_mock.llen.return_value = 0
    redis_mock.lindex.return_value = None
    redis_mock.hget.return_value = None
    redis_mock.delete.return_value = 2
    return redis_mock


@pytest.fixture
def execution_context():
    """Create ExecutionContext for testing."""
    graph = TaskGraph()
    return ExecutionContext(graph)


class TestRedisTaskQueue:
    """Test Redis TaskQueue implementation."""

    def test_redis_not_available(self, execution_context):
        """Test graceful handling when redis is not available."""
        with patch('graflow.queue.redis.redis', None):
            with pytest.raises(ImportError, match="Redis library not installed"):
                from graflow.queue.redis import RedisTaskQueue
                RedisTaskQueue(execution_context)

    @patch('graflow.queue.redis.redis')
    def test_redis_taskqueue_creation_default_client(self, mock_redis_module, execution_context):
        """Test RedisTaskQueue creation with default client."""
        mock_redis_client = Mock()
        mock_redis_module.Redis.return_value = mock_redis_client

        queue = RedisTaskQueue(execution_context)

        assert queue.execution_context == execution_context
        assert queue.redis_client == mock_redis_client
        assert queue.key_prefix == "graflow"
        assert queue.session_id == execution_context.session_id

        # Check Redis keys format
        expected_queue_key = f"graflow:queue:{execution_context.session_id}"
        expected_specs_key = f"graflow:specs:{execution_context.session_id}"
        assert queue.queue_key == expected_queue_key
        assert queue.specs_key == expected_specs_key

        mock_redis_module.Redis.assert_called_once_with(
            host='localhost', port=6379, db=0, decode_responses=True
        )

    def test_redis_taskqueue_creation_custom_client(self, mock_redis, execution_context):
        """Test RedisTaskQueue creation with custom client."""
        with patch('graflow.queue.redis.redis'):
            queue = RedisTaskQueue(execution_context, mock_redis, "test_prefix")

            assert queue.redis_client == mock_redis
            assert queue.key_prefix == "test_prefix"
            assert queue.queue_key.startswith("test_prefix:queue:")
            assert queue.specs_key.startswith("test_prefix:specs:")

    def test_enqueue(self, mock_redis, execution_context):
        """Test enqueuing TaskSpec to Redis."""
        with patch('graflow.queue.redis.redis'):
            queue = RedisTaskQueue(execution_context, mock_redis)

            task_spec = TaskSpec(
                task_id="test_node",
                execution_context=execution_context,
                status=TaskStatus.READY,
                created_at=1234567890.0
            )

            result = queue.enqueue(task_spec)

            assert result is True
            assert queue._task_specs["test_node"] == task_spec

            # Verify Redis calls
            expected_spec_data = {
                'node_id': 'test_node',
                'status': 'ready',
                'created_at': 1234567890.0,
                # Phase 3: Advanced features
                'retry_count': 0,
                'max_retries': 3,
                'last_error': None
            }
            mock_redis.hset.assert_called_once_with(
                queue.specs_key,
                "test_node",
                json.dumps(expected_spec_data)
            )
            mock_redis.rpush.assert_called_once_with(queue.queue_key, "test_node")

    def test_dequeue_empty_queue(self, mock_redis, execution_context):
        """Test dequeue from empty queue."""
        mock_redis.lpop.return_value = None

        with patch('graflow.queue.redis.redis'):
            queue = RedisTaskQueue(execution_context, mock_redis)

            result = queue.dequeue()

            assert result is None
            mock_redis.lpop.assert_called_once_with(queue.queue_key)

    def test_dequeue_missing_spec(self, mock_redis, execution_context):
        """Test dequeue when spec is missing from hash."""
        mock_redis.lpop.return_value = "test_node"
        mock_redis.hget.return_value = None

        with patch('graflow.queue.redis.redis'):
            queue = RedisTaskQueue(execution_context, mock_redis)

            result = queue.dequeue()

            assert result is None
            mock_redis.lpop.assert_called_once_with(queue.queue_key)
            mock_redis.hget.assert_called_once_with(queue.specs_key, "test_node")

    def test_dequeue_success(self, mock_redis, execution_context):
        """Test successful dequeue operation."""
        mock_redis.lpop.return_value = "test_node"
        spec_data = {
            'node_id': 'test_node',
            'status': 'ready',
            'created_at': 1234567890.0
        }
        mock_redis.hget.return_value = json.dumps(spec_data)

        with patch('graflow.queue.redis.redis'):
            queue = RedisTaskQueue(execution_context, mock_redis)

            result = queue.dequeue()

            assert result is not None
            assert result.task_id == "test_node"
            assert result.status == TaskStatus.RUNNING
            assert result.created_at == 1234567890.0
            assert result.execution_context == execution_context
            assert queue._task_specs["test_node"] == result

            mock_redis.lpop.assert_called_once_with(queue.queue_key)
            mock_redis.hget.assert_called_once_with(queue.specs_key, "test_node")

    def test_is_empty(self, mock_redis, execution_context):
        """Test is_empty method."""
        with patch('graflow.queue.redis.redis'):
            queue = RedisTaskQueue(execution_context, mock_redis)

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
            queue = RedisTaskQueue(execution_context, mock_redis)

            result = queue.size()

            assert result == 5
            mock_redis.llen.assert_called_once_with(queue.queue_key)

    def test_peek_next_node(self, mock_redis, execution_context):
        """Test peek_next_node method."""
        with patch('graflow.queue.redis.redis'):
            queue = RedisTaskQueue(execution_context, mock_redis)

            # Test empty queue
            mock_redis.lindex.return_value = None
            result = queue.peek_next_node()
            assert result is None

            # Test queue with items
            mock_redis.lindex.return_value = "test_node"
            result = queue.peek_next_node()
            assert result == "test_node"

            # Verify Redis calls
            assert mock_redis.lindex.call_count == 2
            mock_redis.lindex.assert_called_with(queue.queue_key, 0)

    def test_cleanup(self, mock_redis, execution_context):
        """Test cleanup method."""
        with patch('graflow.queue.redis.redis'):
            queue = RedisTaskQueue(execution_context, mock_redis)

            queue.cleanup()

            mock_redis.delete.assert_called_once_with(queue.queue_key, queue.specs_key)

    def test_legacy_api_compatibility(self, mock_redis, execution_context):
        """Test legacy API methods work with Redis backend."""
        with patch('graflow.queue.redis.redis'):
            queue = RedisTaskQueue(execution_context, mock_redis)

            # Test add_node
            queue.add_node("test_node")

            # Verify enqueue was called via add_node
            mock_redis.hset.assert_called_once()
            mock_redis.rpush.assert_called_once_with(queue.queue_key, "test_node")

            # Test get_next_node (empty queue)
            mock_redis.lpop.return_value = None
            result = queue.get_next_node()
            assert result is None


class TestRedisTaskQueueIntegration:
    """Integration tests for Redis TaskQueue with ExecutionContext."""

    def test_execution_context_with_redis_backend(self, mock_redis):
        """Test ExecutionContext integration with Redis backend."""
        graph = TaskGraph()

        with patch('graflow.queue.redis.redis'):
            # Test Redis backend creation
            context = ExecutionContext(
                graph,
                start_node="start",
                queue_backend=QueueBackend.REDIS,
                queue_config={
                    'redis_client': mock_redis,
                    'key_prefix': 'test'
                }
            )

            assert isinstance(context.task_queue, RedisTaskQueue)
            assert context.task_queue.redis_client == mock_redis
            assert context.task_queue.key_prefix == 'test'

    def test_execution_context_redis_string_backend(self, mock_redis):
        """Test ExecutionContext with redis string backend."""
        graph = TaskGraph()

        with patch('graflow.queue.redis.redis'):
            context = ExecutionContext(
                graph,
                start_node="start",
                queue_backend="redis",
                queue_config={'redis_client': mock_redis}
            )

            assert isinstance(context.task_queue, RedisTaskQueue)

    def test_redis_backend_compatibility_methods(self, mock_redis):
        """Test that Redis backend maintains compatibility with ExecutionContext."""
        graph = TaskGraph()

        # Configure mock to return proper data for dequeue
        spec_data = {
            'node_id': 'start',
            'status': 'ready',
            'created_at': time.time()
        }
        mock_redis.hget.return_value = json.dumps(spec_data)

        with patch('graflow.queue.redis.redis'):
            context = ExecutionContext(
                graph,
                start_node="start",
                queue_backend=QueueBackend.REDIS,
                queue_config={'redis_client': mock_redis}
            )

            # Test compatibility methods
            context.add_to_queue("task1")
            context.add_to_queue("task2")

            # Mock dequeue responses - lpop returns node_ids, hget returns spec data
            mock_redis.lpop.side_effect = ["start", "task1", "task2", None]

            # Setup hget to return appropriate spec data for each node
            def mock_hget(specs_key, node_id):
                spec_data = {
                    'node_id': node_id,
                    'status': 'ready',
                    'created_at': time.time()
                }
                return json.dumps(spec_data)
            mock_redis.hget.side_effect = mock_hget

            # Test getting nodes
            node1 = context.get_next_node()
            node2 = context.get_next_node()
            node3 = context.get_next_node()
            node4 = context.get_next_node()

            assert node1 == "start"
            assert node2 == "task1"
            assert node3 == "task2"
            assert node4 is None

            # Test is_completed
            mock_redis.llen.return_value = 0
            assert context.is_completed() is True
