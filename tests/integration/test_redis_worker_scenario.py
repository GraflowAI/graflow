"""Integration tests for Redis-backed TaskWorker scenarios."""

import time
from typing import Any, Callable, Dict, Optional, TypedDict

import pytest

from graflow.channels.redis import RedisChannel
from graflow.channels.schemas import TaskResultMessage
from graflow.coordination.redis import RedisCoordinator
from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import Executable
from graflow.queue.base import TaskSpec, TaskStatus
from graflow.queue.distributed import DistributedTaskQueue
from graflow.worker.worker import TaskWorker

CHANNEL_NAME = "task_channel"
_redis_client: Optional[Any] = None


class DataMessage(TypedDict):
    """Custom message type for data exchange."""
    data: list[int]
    task_id: str
    timestamp: float


class RedisWorkflowTask(Executable):
    """Executable wrapper for Redis-backed tests."""

    def __init__(self, task_id: str, func: Callable[[], Any]) -> None:
        super().__init__()
        self._task_id = task_id
        self._func = func

    @property
    def task_id(self) -> str:
        return self._task_id

    def run(self) -> Any:
        return self._func()

    def __call__(self) -> Any:
        return self.run()


def _configure_redis_client(redis_client: Any) -> None:
    """Set the Redis client used by task helpers."""
    global _redis_client
    _redis_client = redis_client


def _get_channel() -> RedisChannel:
    """Create a RedisChannel bound to the configured client."""
    if _redis_client is None:
        raise RuntimeError("Redis client not configured for integration tasks")
    return RedisChannel(CHANNEL_NAME, redis_client=_redis_client)


def _register_tasks(execution_context: ExecutionContext, tasks: Dict[str, RedisWorkflowTask]) -> None:
    """Add task executables to the execution graph.

    Tasks are stored in Graph, no separate registration needed.
    """
    for task in tasks.values():
        if task.task_id not in execution_context.graph.nodes:
            execution_context.graph.add_node(task, task.task_id)


def _make_task_spec(task: RedisWorkflowTask, execution_context: ExecutionContext) -> TaskSpec:
    """Create TaskSpec for distributed execution.

    Tasks are retrieved from Graph (via GraphStore), no explicit
    serialization strategy needed.
    """
    return TaskSpec(
        executable=task,
        execution_context=execution_context,
        status=TaskStatus.READY
    )


def _create_producer_task(task_id: str, data: list[int]) -> RedisWorkflowTask:
    """Create a producer task that writes data to the shared Redis channel."""

    def run() -> str:
        channel = _get_channel()
        data_msg: DataMessage = {
            "data": data,
            "task_id": task_id,
            "timestamp": time.time()
        }
        channel.set(f"data:{task_id}", data_msg)

        counter_raw = channel.get("producer_count", 0)
        counter = counter_raw if isinstance(counter_raw, int) else 0
        channel.set("producer_count", counter + 1)
        return f"produced: {data}"

    return RedisWorkflowTask(task_id, run)


def _create_consumer_task(task_id: str) -> RedisWorkflowTask:
    """Create a consumer task that aggregates producer data."""

    def run() -> str:
        channel = _get_channel()
        timeout = time.time() + 5.0
        while time.time() < timeout:
            counter_raw = channel.get("producer_count", 0)
            producer_count = counter_raw if isinstance(counter_raw, int) else 0
            if producer_count >= 2:
                break
            time.sleep(0.1)

        consumed_data: list[int] = []
        for key in channel.keys():
            if key.startswith("data:producer"):
                data_msg = channel.get(key)
                if isinstance(data_msg, dict):
                    consumed_data.extend(data_msg.get("data", []))

        result = f"consumed_{task_id}: {consumed_data}"
        result_msg: TaskResultMessage = {
            "task_id": task_id,
            "result": consumed_data,
            "timestamp": time.time(),
            "status": "completed"
        }
        channel.set(f"consumed:{task_id}", result_msg)
        return result

    return RedisWorkflowTask(task_id, run)


def _create_aggregator_task(task_id: str) -> RedisWorkflowTask:
    """Create an aggregator task that composes consumer results."""

    def run() -> str:
        channel = _get_channel()
        timeout = time.time() + 5.0
        aggregated_results: list[int] = []

        while time.time() < timeout:
            consumed_keys = [key for key in channel.keys() if key.startswith("consumed:")]
            if len(consumed_keys) >= 2:
                for key in consumed_keys:
                    result_data = channel.get(key)
                    if isinstance(result_data, dict):
                        aggregated_results.extend(result_data.get("result", []))
                break
            time.sleep(0.1)

        final_result = f"aggregated: {len(aggregated_results)} items total"
        final_msg: TaskResultMessage = {
            "task_id": task_id,
            "result": final_result,
            "timestamp": time.time(),
            "status": "completed"
        }
        channel.set("final_result", final_msg)
        return final_result

    return RedisWorkflowTask(task_id, run)


def _create_simple_task(task_id: str) -> RedisWorkflowTask:
    """Create a task that records its execution in Redis."""

    def run() -> str:
        channel = _get_channel()
        channel.set(f"simple:{task_id}", f"executed_{task_id}")
        return f"completed_{task_id}"

    return RedisWorkflowTask(task_id, run)


@pytest.fixture
def execution_context(clean_redis):
    """Create real ExecutionContext with Redis channel support."""
    graph = TaskGraph()
    context = ExecutionContext(
        graph,
        channel_backend="redis",
        config={"redis_client": clean_redis, "key_prefix": "test_graflow"}
    )
    _configure_redis_client(clean_redis)
    # Ensure channel state is clean
    RedisChannel(CHANNEL_NAME, redis_client=clean_redis).clear()
    return context


@pytest.fixture
def redis_queue(clean_redis, execution_context):
    """Create RedisTaskQueue bound to the execution context."""
    queue = DistributedTaskQueue(redis_client=clean_redis, key_prefix="test_graflow")
    queue.cleanup()
    yield queue
    queue.cleanup()


@pytest.fixture
def redis_coordinator(redis_queue):
    """Create RedisCoordinator backed by the test RedisTaskQueue."""
    return RedisCoordinator(redis_queue)


@pytest.fixture
def task_worker(redis_queue):
    """Create TaskWorker instance for tests."""
    worker = TaskWorker(
        queue=redis_queue,
        worker_id="test_worker",
        max_concurrent_tasks=3,
        poll_interval=0.1
    )
    yield worker
    if worker.is_running:
        worker.stop()


def test_producer_consumer_with_typed_channels(redis_queue, task_worker, clean_redis, execution_context):
    """Test producer-consumer pattern using typed channels inside tasks."""

    tasks = {
        "producer_task_0": _create_producer_task("producer_task_0", [1, 2, 3, 4, 5]),
        "producer_task_1": _create_producer_task("producer_task_1", [6, 7, 8, 9, 10]),
        "consumer_task_0": _create_consumer_task("consumer_task_0"),
        "consumer_task_1": _create_consumer_task("consumer_task_1"),
        "aggregator_task": _create_aggregator_task("aggregator_task")
    }
    _register_tasks(execution_context, tasks)

    # Enqueue tasks in desired execution order
    enqueue_order = [
        "producer_task_0",
        "producer_task_1",
        "consumer_task_0",
        "consumer_task_1",
        "aggregator_task"
    ]
    for task_id in enqueue_order:
        redis_queue.enqueue(_make_task_spec(tasks[task_id], execution_context))

    # Start worker
    task_worker.start()

    # Wait for all tasks to complete
    timeout = 10.0
    start_time = time.time()
    while redis_queue.size() > 0 and time.time() - start_time < timeout:
        time.sleep(0.2)

    time.sleep(1.0)  # Allow final processing
    task_worker.stop()

    # Verify the pipeline worked using direct Redis access
    redis_channel = RedisChannel("task_channel", host="localhost", port=6379, db=0)
    redis_channel.redis_client = clean_redis

    # Check final result exists
    assert redis_channel.exists("final_result")
    final_result = redis_channel.get("final_result")
    assert isinstance(final_result, dict)
    assert final_result["status"] == "completed"
    assert "aggregated:" in final_result["result"]

    # Verify producer counter
    assert redis_channel.get("producer_count", 0) >= 2

    # Verify worker metrics
    metrics = task_worker.get_metrics()
    assert metrics["tasks_processed"] == 5
    assert metrics["tasks_succeeded"] == 5


def test_parallel_coordination_with_redis(redis_queue, clean_redis, redis_coordinator, execution_context):
    """Test parallel task coordination using Redis channels and barriers."""

    # Create multiple workers
    workers = []
    for i in range(2):
        worker = TaskWorker(
            queue=redis_queue,
            worker_id=f"worker_{i}",
            max_concurrent_tasks=2,
            poll_interval=0.1
        )
        workers.append(worker)

    try:
        # Create coordination barrier
        barrier_id = "task_sync"
        redis_coordinator.create_barrier(barrier_id, 4)  # 4 parallel tasks

        tasks = {
            f"simple_task_{i}": _create_simple_task(f"simple_task_{i}")
            for i in range(4)
        }
        _register_tasks(execution_context, tasks)

        # Create tasks
        for i in range(4):
            redis_queue.enqueue(_make_task_spec(tasks[f"simple_task_{i}"], execution_context))

        # Start workers
        for worker in workers:
            worker.start()

        # Wait for processing
        timeout = 8.0
        start_time = time.time()
        while redis_queue.size() > 0 and time.time() - start_time < timeout:
            time.sleep(0.2)

        time.sleep(1.0)

        # Verify all tasks used the channel
        redis_channel = RedisChannel("task_channel", host="localhost", port=6379, db=0)
        redis_channel.redis_client = clean_redis

        simple_keys = [k for k in redis_channel.keys() if k.startswith("simple:")]
        assert len(simple_keys) == 4

        # Verify all tasks executed
        total_processed = sum(worker.get_metrics()["tasks_processed"] for worker in workers)
        assert total_processed == 4

    finally:
        for worker in workers:
            if worker.is_running:
                worker.stop()


def test_redis_channel_operations_in_tasks(clean_redis):
    """Test Redis channel operations as used by tasks."""

    # Create channel like tasks would
    channel = RedisChannel("task_channel", host="localhost", port=6379, db=0)
    channel.redis_client = clean_redis

    # Test message passing pattern used by tasks
    data_msg: DataMessage = {
        "data": [1, 2, 3],
        "task_id": "test_task",
        "timestamp": time.time()
    }

    # Producer writes data
    channel.set("data:test_task", data_msg)

    # Consumer reads data
    received_msg = channel.get("data:test_task")
    assert received_msg["data"] == [1, 2, 3]
    assert received_msg["task_id"] == "test_task"

    # Result storage
    result_msg: TaskResultMessage = {
        "task_id": "test_task",
        "result": "processed_data",
        "timestamp": time.time(),
        "status": "completed"
    }

    channel.set("result:test_task", result_msg)
    stored_result = channel.get("result:test_task")
    assert stored_result["status"] == "completed"
    assert stored_result["result"] == "processed_data"
