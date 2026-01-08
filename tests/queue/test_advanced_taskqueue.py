"""Tests for advanced TaskQueue features (Phase 3)."""

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import Task
from graflow.queue.base import TaskSpec, TaskStatus
from graflow.queue.local import LocalTaskQueue


@pytest.fixture
def execution_context():
    """Create ExecutionContext for testing."""
    graph = TaskGraph()
    return ExecutionContext(graph)


class TestTaskSpecAdvancedFeatures:
    """Test TaskSpec advanced features."""

    def test_task_spec_retry_fields(self, execution_context: ExecutionContext):
        """Test TaskSpec retry-related fields."""
        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False), execution_context=execution_context, max_retries=5
        )

        assert task_spec.retry_count == 0
        assert task_spec.max_retries == 5
        assert task_spec.last_error is None
        assert task_spec.can_retry() is True

    def test_task_spec_can_retry(self, execution_context: ExecutionContext):
        """Test can_retry logic."""
        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False), execution_context=execution_context, max_retries=2
        )

        # Can retry initially
        assert task_spec.can_retry() is True

        # After first retry
        task_spec.increment_retry("First error")
        assert task_spec.retry_count == 1
        assert task_spec.can_retry() is True
        assert task_spec.last_error == "First error"

        # After second retry
        task_spec.increment_retry("Second error")
        assert task_spec.retry_count == 2
        assert task_spec.can_retry() is False
        assert task_spec.last_error == "Second error"

    def test_task_spec_increment_retry(self, execution_context: ExecutionContext):
        """Test increment_retry behavior."""
        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False),
            execution_context=execution_context,
            status=TaskStatus.ERROR,
        )

        task_spec.increment_retry("Test error")

        assert task_spec.retry_count == 1
        assert task_spec.last_error == "Test error"
        assert task_spec.status == TaskStatus.READY  # Reset for retry


class TestAbstractTaskQueueAdvancedFeatures:
    """Test AbstractTaskQueue advanced features."""

    def test_queue_configuration(self, execution_context: ExecutionContext):
        """Test queue configuration."""
        queue = LocalTaskQueue(execution_context)

        # Default configuration
        assert queue.enable_retry is False
        assert queue.enable_metrics is False

        # Configure advanced features
        queue.configure(enable_retry=True, enable_metrics=True)

        assert queue.enable_retry is True
        assert queue.enable_metrics is True

    def test_metrics_initialization(self, execution_context: ExecutionContext):
        """Test metrics initialization."""
        queue = LocalTaskQueue(execution_context)

        metrics = queue.get_metrics()
        expected_metrics = {"enqueued": 0, "dequeued": 0, "retries": 0, "failures": 0}

        assert metrics == expected_metrics

    def test_handle_task_failure_without_retry(self, execution_context: ExecutionContext):
        """Test task failure handling without retry enabled."""
        queue = LocalTaskQueue(execution_context)
        queue.configure(enable_retry=False, enable_metrics=True)

        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False), execution_context=execution_context
        )

        should_retry = queue.handle_task_failure(task_spec, "Test error")

        assert should_retry is False
        assert task_spec.status == TaskStatus.ERROR
        assert queue.get_metrics()["failures"] == 1

    def test_handle_task_failure_with_retry(self, execution_context: ExecutionContext):
        """Test task failure handling with retry enabled."""
        queue = LocalTaskQueue(execution_context)
        queue.configure(enable_retry=True, enable_metrics=True)

        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False), execution_context=execution_context, max_retries=2
        )

        # First failure - should retry
        should_retry = queue.handle_task_failure(task_spec, "First error")

        assert should_retry is True
        assert task_spec.retry_count == 1
        assert task_spec.last_error == "First error"
        assert task_spec.status == TaskStatus.READY
        assert queue.get_metrics()["failures"] == 1
        assert queue.get_metrics()["retries"] == 1

    def test_handle_task_failure_max_retries_exceeded(self, execution_context: ExecutionContext):
        """Test task failure when max retries exceeded."""
        queue = LocalTaskQueue(execution_context)
        queue.configure(enable_retry=True, enable_metrics=True)

        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False),
            execution_context=execution_context,
            max_retries=1,
            retry_count=1,  # Already at max
        )

        should_retry = queue.handle_task_failure(task_spec, "Final error")

        assert should_retry is False
        assert task_spec.status == TaskStatus.ERROR
        assert queue.get_metrics()["failures"] == 1
        assert queue.get_metrics()["retries"] == 0  # No more retries

    def test_reset_metrics(self, execution_context: ExecutionContext):
        """Test metrics reset."""
        queue = LocalTaskQueue(execution_context)
        queue.configure(enable_metrics=True)

        # Simulate some activity
        queue.metrics["enqueued"] = 5
        queue.metrics["failures"] = 2

        queue.reset_metrics()

        expected_metrics = {"enqueued": 0, "dequeued": 0, "retries": 0, "failures": 0}

        assert queue.get_metrics() == expected_metrics


class TestInMemoryTaskQueueAdvancedFeatures:
    """Test InMemoryTaskQueue advanced features."""

    def test_enqueue_with_metrics(self, execution_context: ExecutionContext):
        """Test enqueue with metrics enabled."""
        queue = LocalTaskQueue(execution_context)
        queue.configure(enable_metrics=True)

        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False), execution_context=execution_context
        )

        queue.enqueue(task_spec)

        assert queue.get_metrics()["enqueued"] == 1

    def test_dequeue_with_metrics(self, execution_context: ExecutionContext):
        """Test dequeue with metrics enabled."""
        queue = LocalTaskQueue(execution_context)
        queue.configure(enable_metrics=True)

        # Enqueue a task first
        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False), execution_context=execution_context
        )
        queue.enqueue(task_spec)

        assert queue.get_metrics()["dequeued"] == 0

        # Now dequeue it
        dequeued_spec = queue.dequeue()

        assert dequeued_spec is not None
        assert queue.get_metrics()["dequeued"] == 1

    def test_retry_failed_task(self, execution_context: ExecutionContext):
        """Test retry_failed_task method."""
        queue = LocalTaskQueue(execution_context)
        queue.configure(enable_retry=True, enable_metrics=True)

        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False),
            execution_context=execution_context,
            status=TaskStatus.ERROR,
            retry_count=1,
            max_retries=3,
        )

        # Should succeed
        result = queue.retry_failed_task(task_spec)

        assert result is True
        assert task_spec.status == TaskStatus.READY
        assert queue.size() == 1
        assert queue.get_metrics()["enqueued"] == 1

    def test_retry_failed_task_max_retries_exceeded(self, execution_context: ExecutionContext):
        """Test retry when max retries exceeded."""
        queue = LocalTaskQueue(execution_context)
        queue.configure(enable_retry=True)

        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False),
            execution_context=execution_context,
            status=TaskStatus.ERROR,
            retry_count=3,
            max_retries=3,
        )

        # Should fail
        result = queue.retry_failed_task(task_spec)

        assert result is False
        assert queue.size() == 0

    def test_retry_failed_task_retry_disabled(self, execution_context: ExecutionContext):
        """Test retry when retry is disabled."""
        queue = LocalTaskQueue(execution_context)
        queue.configure(enable_retry=False)

        task_spec = TaskSpec(
            executable=Task("test_task", register_to_context=False),
            execution_context=execution_context,
            status=TaskStatus.ERROR,
        )

        # Should fail
        result = queue.retry_failed_task(task_spec)

        assert result is False
        assert queue.size() == 0


class TestExecutionContextAdvancedFeatures:
    """Test ExecutionContext integration with advanced features."""

    def test_execution_context_with_advanced_queue_config(self):
        """Test ExecutionContext with advanced queue configuration."""
        graph = TaskGraph()

        context = ExecutionContext.create(
            graph,
            "start",
        )

        # Configure queue for advanced features
        if hasattr(context.task_queue, "configure"):
            context.task_queue.configure(enable_retry=True, enable_metrics=True)

        assert context.task_queue.enable_retry is True
        assert context.task_queue.enable_metrics is True

        # Test metrics are updated
        task1 = Task("task1", register_to_context=False)
        context.add_to_queue(task1)
        assert context.task_queue.get_metrics()["enqueued"] == 1
