"""Pytest-style tests for coordination backends."""

import json
from types import SimpleNamespace
from unittest.mock import call

import pytest

from graflow.coordination.coordinator import CoordinationBackend
from graflow.coordination.redis import RedisCoordinator, record_task_completion
from graflow.coordination.threading import ThreadingCoordinator
from graflow.queue.distributed import DistributedTaskQueue


class TestThreadingCoordinator:
    """Test cases for ThreadingCoordinator."""

    @pytest.fixture
    def coordinator(self):
        """Create a coordinator with a predictable thread count."""
        coordinator = ThreadingCoordinator(thread_count=2)
        yield coordinator
        coordinator.shutdown()

    def test_threading_coordinator_creation(self, coordinator):
        """Thread count is stored and executor starts lazily."""
        assert coordinator.thread_count == 2
        assert coordinator._executor is None

    def test_threading_coordinator_thread_count(self):
        """Explicit thread count is respected."""
        coord = ThreadingCoordinator(thread_count=4)
        try:
            assert coord.thread_count == 4
        finally:
            coord.shutdown()

    def test_threading_coordinator_default_thread_count(self):
        """Default thread count derives from CPU count."""
        coord = ThreadingCoordinator()
        try:
            assert coord.thread_count > 0
        finally:
            coord.shutdown()

    def test_execute_group_runs_all_tasks(self, coordinator, mocker):
        """All tasks are executed through the workflow engine."""
        exec_context = mocker.Mock(name="execution_context")
        created_branch_contexts: dict[str, object] = {}

        def fake_create_branch_context(*, branch_id: str):
            branch_context = mocker.Mock(name=f"branch_context_{branch_id}")
            branch_context.session_id = f"session-{branch_id}"
            created_branch_contexts[branch_id] = branch_context
            return branch_context

        exec_context.create_branch_context.side_effect = fake_create_branch_context
        tasks = [SimpleNamespace(task_id="task1"), SimpleNamespace(task_id="task2")]
        engine_calls: list[tuple[object, str]] = []

        def engine_factory():
            def execute(context, start_task_id):
                engine_calls.append((context, start_task_id))
            return SimpleNamespace(execute=execute)

        mocker.patch("graflow.core.engine.WorkflowEngine", side_effect=engine_factory)

        mock_handler = mocker.Mock(name="handler")

        coordinator.execute_group("group-a", tasks, exec_context, mock_handler)

        assert exec_context.create_branch_context.call_count == len(tasks)
        assert len(engine_calls) == len(tasks)
        assert {task_id for _, task_id in engine_calls} == {"task1", "task2"}
        branch_contexts_used = {context for context, _ in engine_calls}
        assert branch_contexts_used == set(created_branch_contexts.values())

    def test_execute_group_handles_task_failure(self, coordinator, mocker):
        """Failures from the workflow engine are reported but do not stop execution."""
        exec_context = mocker.Mock(name="execution_context")

        def fake_create_branch_context(*, branch_id: str):
            branch_context = mocker.Mock(name=f"branch_context_{branch_id}")
            branch_context.session_id = f"session-{branch_id}"
            return branch_context

        exec_context.create_branch_context.side_effect = fake_create_branch_context
        exec_context.merge_results = mocker.Mock()
        exec_context.mark_branch_completed = mocker.Mock()

        tasks = [SimpleNamespace(task_id="task_fail"), SimpleNamespace(task_id="task_ok")]
        completed = []

        def engine_factory():
            def execute(context, start_task_id):
                if start_task_id == "task_fail":
                    raise RuntimeError("boom")
                completed.append(start_task_id)
            return SimpleNamespace(execute=execute)

        mocker.patch("graflow.core.engine.WorkflowEngine", side_effect=engine_factory)

        mock_handler = mocker.Mock(name="handler")

        # The test expects the coordinator to continue execution despite failures
        coordinator.execute_group("group-b", tasks, exec_context, mock_handler)

        # Verify that task_ok completed (failure didn't stop execution)
        assert "task_ok" in completed

    def test_shutdown(self):
        """Executor shuts down and clears internal reference."""
        coord = ThreadingCoordinator(thread_count=2)
        try:
            coord._ensure_executor()
            assert coord._executor is not None
        finally:
            coord.shutdown()
        assert coord._executor is None


class TestRedisCoordinator:
    """Test cases for RedisCoordinator."""

    @pytest.fixture
    def mock_task_queue(self, mocker):
        """Provide a RedisTaskQueue mock with a redis client."""
        queue = mocker.Mock(spec=DistributedTaskQueue)
        queue.redis_client = mocker.Mock()
        queue.key_prefix = "test"
        queue.queue_key = f"{queue.key_prefix}:queue"
        queue.graph_store = None
        return queue

    @pytest.fixture
    def coordinator(self, mock_task_queue):
        """Coordinator under test."""
        return RedisCoordinator(mock_task_queue)

    @pytest.fixture
    def mock_redis(self, mock_task_queue):
        """Shortcut to the mocked redis client."""
        return mock_task_queue.redis_client

    def test_redis_coordinator_creation(self, coordinator, mock_task_queue, mock_redis):
        """Coordinator stores queue and redis references."""
        assert coordinator.task_queue is mock_task_queue
        assert coordinator.redis is mock_redis
        assert coordinator.active_barriers == {}

    def test_create_barrier(self, coordinator, mock_redis):
        """Barrier metadata is stored and redis keys are prepared."""
        barrier_key = coordinator.create_barrier("test_barrier", 3)

        assert barrier_key == f"{coordinator.task_queue.key_prefix}:barrier:test_barrier"
        assert "test_barrier" in coordinator.active_barriers
        barrier_info = coordinator.active_barriers["test_barrier"]
        assert barrier_info["expected"] == 3
        assert barrier_info["current"] == 0

        mock_redis.delete.assert_called_with(f"{coordinator.task_queue.key_prefix}:barrier:test_barrier")
        mock_redis.set.assert_called_with(f"{coordinator.task_queue.key_prefix}:barrier:test_barrier:expected", 3)

    def test_wait_barrier_success(self, coordinator, mock_redis):
        """Last participant publishes completion event."""
        coordinator.create_barrier("test_barrier", 2)
        mock_redis.incr.return_value = 2

        result = coordinator.wait_barrier("test_barrier", timeout=1)

        assert result is True
        mock_redis.incr.assert_called_with(f"{coordinator.task_queue.key_prefix}:barrier:test_barrier")
        mock_redis.publish.assert_called_with(
            f"{coordinator.task_queue.key_prefix}:barrier_done:test_barrier", "complete"
        )

    def test_wait_barrier_with_pubsub(self, coordinator, mock_redis, mocker):
        """Participants block on pub/sub until completion message arrives."""
        coordinator.create_barrier("test_barrier", 2)
        mock_redis.incr.return_value = 1

        mock_pubsub = mocker.MagicMock()
        mock_pubsub.listen.return_value = iter([{ "type": "message", "data": b"complete" }])
        mock_redis.pubsub.return_value = mock_pubsub

        result = coordinator.wait_barrier("test_barrier", timeout=1)

        assert result is True
        mock_pubsub.subscribe.assert_called_with(
            f"{coordinator.task_queue.key_prefix}:barrier_done:test_barrier"
        )
        mock_pubsub.close.assert_called_once()

    def test_wait_barrier_timeout(self, coordinator, mock_redis, mocker):
        """Timeout returns False and closes pubsub."""
        coordinator.create_barrier("timeout_barrier", 2)
        mock_redis.incr.return_value = 1

        mock_pubsub = mocker.MagicMock()

        def listen():
            while True:
                yield {"type": "message", "data": b"pending"}

        mock_pubsub.listen.return_value = listen()
        mock_redis.pubsub.return_value = mock_pubsub

        mocker.patch("graflow.coordination.redis.time.time", side_effect=[0, 0.5, 1.5])

        result = coordinator.wait_barrier("timeout_barrier", timeout=1)

        assert result is False
        mock_pubsub.close.assert_called_once()

    def test_wait_barrier_nonexistent(self, coordinator):
        """Waiting on a missing barrier returns False."""
        assert coordinator.wait_barrier("missing", timeout=1) is False

    def test_dispatch_task_serializes_record(self, coordinator, mock_task_queue, mocker):
        """Dispatch pushes SerializedTaskRecord into Redis with graph hash."""
        exec_context = mocker.Mock()
        exec_context.session_id = "sess-1"
        exec_context.trace_id = "trace-1"
        exec_context.graph_hash = "graph-123"
        exec_context.span_id = None
        executable = mocker.Mock()
        executable.task_id = "task-123"
        executable.get_execution_context.return_value = exec_context

        coordinator.dispatch_task(executable, "group-1")

        mock_task_queue.redis_client.lpush.assert_called_once()
        queue_key, payload = mock_task_queue.redis_client.lpush.call_args[0]
        assert queue_key == mock_task_queue.queue_key
        record = json.loads(payload)
        assert record["task_id"] == "task-123"
        assert record["group_id"] == "group-1"
        assert record["graph_hash"] == "graph-123"
        assert record["session_id"] == "sess-1"

    def test_execute_group_success_flow(self, coordinator, mocker):
        """Parallel execution creates barrier, dispatches tasks, waits, and cleans up."""
        tasks = [SimpleNamespace(task_id="task1"), SimpleNamespace(task_id="task2")]

        mock_create = mocker.patch.object(
            coordinator,
            "create_barrier",
            return_value=f"{coordinator.task_queue.key_prefix}:barrier:test_group",
        )
        mock_dispatch = mocker.patch.object(coordinator, "dispatch_task")
        mock_wait = mocker.patch.object(coordinator, "wait_barrier", return_value=True)
        mock_cleanup = mocker.patch.object(coordinator, "cleanup_barrier")
        mock_results = mocker.patch.object(coordinator, "_get_completion_results", return_value=[])
        mock_handler = mocker.Mock(name="handler")

        coordinator.execute_group("test_group", tasks, mocker.Mock(), mock_handler)

        mock_create.assert_called_once_with("test_group", len(tasks))
        mock_dispatch.assert_has_calls([call(tasks[0], "test_group"), call(tasks[1], "test_group")])
        mock_wait.assert_called_once_with("test_group")
        mock_results.assert_called_once_with("test_group")
        mock_cleanup.assert_called_once_with("test_group")

    def test_execute_group_timeout_triggers_cleanup(self, coordinator, mocker):
        """TimeoutError is raised and barrier cleanup still executes."""
        tasks = [SimpleNamespace(task_id="task1")]

        mocker.patch.object(
            coordinator,
            "create_barrier",
            return_value=f"{coordinator.task_queue.key_prefix}:barrier:test_group",
        )
        mocker.patch.object(coordinator, "dispatch_task")
        mock_wait = mocker.patch.object(coordinator, "wait_barrier", return_value=False)
        mock_cleanup = mocker.patch.object(coordinator, "cleanup_barrier")
        mock_handler = mocker.Mock(name="handler")

        with pytest.raises(TimeoutError):
            coordinator.execute_group("test_group", tasks, mocker.Mock(), mock_handler)

        mock_wait.assert_called_once_with("test_group")
        mock_cleanup.assert_called_once_with("test_group")

    def test_cleanup_barrier(self, coordinator, mock_redis):
        """Cleanup removes redis keys and active barrier entry."""
        coordinator.create_barrier("cleanup_barrier", 2)
        mock_redis.delete.reset_mock()

        coordinator.cleanup_barrier("cleanup_barrier")

        expected_calls = [
            call(f"{coordinator.task_queue.key_prefix}:barrier:cleanup_barrier"),
            call(f"{coordinator.task_queue.key_prefix}:barrier:cleanup_barrier:expected")
        ]
        mock_redis.delete.assert_has_calls(expected_calls, any_order=True)
        assert "cleanup_barrier" not in coordinator.active_barriers

    def test_queue_operations(self, coordinator, mock_task_queue):
        """Queue helper methods delegate to task queue."""
        mock_task_queue.size.return_value = 5

        assert coordinator.get_queue_size("group-x") == 5
        mock_task_queue.size.assert_called_once_with()

        coordinator.clear_queue("group-x")
        mock_task_queue.cleanup.assert_called_once_with()

    def test_record_task_completion_signals_when_expected_met(self, mocker):
        """record_task_completion publishes completion when all participants finish."""
        redis_client = mocker.Mock()
        redis_client.incr.return_value = 3
        redis_client.get.return_value = b"3"

        record_task_completion(redis_client, "prefix", "task", "group", True)

        redis_client.incr.assert_called_with("prefix:barrier:group")
        redis_client.publish.assert_called_with("prefix:barrier_done:group", "complete")


class TestCoordinationBackend:
    """Test cases for CoordinationBackend enum."""

    def test_coordination_backend_values(self):
        """Enum values remain stable."""
        assert CoordinationBackend.REDIS.value == "redis"
        assert CoordinationBackend.THREADING.value == "threading"
        assert CoordinationBackend.DIRECT.value == "direct"
