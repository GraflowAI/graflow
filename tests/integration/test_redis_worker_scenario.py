"""Integration test for TaskWorker, RedisQueue, InProcessTaskExecutor with Redis channels used inside tasks."""

import time
from typing import TypedDict
from unittest.mock import Mock

import pytest

from graflow.channels.redis import RedisChannel
from graflow.channels.schemas import TaskResultMessage
from graflow.coordination.redis import RedisCoordinator
from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.queue.base import TaskSpec, TaskStatus
from graflow.queue.redis import RedisTaskQueue
from graflow.worker.handler import InProcessTaskExecutor
from graflow.worker.worker import TaskWorker


class DataMessage(TypedDict):
    """Custom message type for data exchange."""
    data: list[int]
    task_id: str
    timestamp: float


@pytest.fixture
def execution_context():
    """Create a mock execution context."""
    context = Mock(spec=ExecutionContext)
    context.session_id = "test_session"
    return context


@pytest.fixture
def redis_queue(clean_redis, execution_context):
    """Create RedisTaskQueue with clean Redis."""
    return RedisTaskQueue(execution_context, clean_redis, "test_graflow")


@pytest.fixture
def redis_coordinator(clean_redis):
    """Create RedisCoordinator."""
    return RedisCoordinator(clean_redis)


class ChannelAwareTaskHandler(InProcessTaskExecutor):
    """Task handler that creates tasks with channel access."""

    def __init__(self, redis_client):
        super().__init__()
        self.redis_client = redis_client

    def _resolve_task(self, task_spec):
        """Create task with proper TaskExecutionContext and channel access."""
        task_id = task_spec.node_id

        # Create mock TaskExecutionContext with Redis channel
        task_context = Mock(spec=TaskExecutionContext)
        task_context.task_id = task_id

        # Create channel that uses the same Redis instance
        redis_channel = RedisChannel(
            name="task_channel",
            host="localhost",
            port=6379,
            db=0  # Use same db as clean_redis
        )
        redis_channel.redis_client = self.redis_client

        # Mock get_typed_channel to return our Redis channel
        def get_typed_channel(message_type):
            return redis_channel

        task_context.get_typed_channel = get_typed_channel
        task_context.get_channel = lambda: redis_channel

        class ChannelTask:
            def __init__(self, task_id: str, ctx: TaskExecutionContext):
                self.task_id = task_id
                self.ctx = ctx

            def __call__(self):
                if self.task_id.startswith("producer"):
                    return self.produce_data()
                elif self.task_id.startswith("consumer"):
                    return self.consume_data()
                elif self.task_id.startswith("aggregator"):
                    return self.aggregate_data()
                else:
                    return self.simple_task()

            def produce_data(self):
                """Producer task using typed channels."""
                # Get typed channel for data exchange
                data_channel = self.ctx.get_typed_channel(DataMessage)

                # Generate and send data
                data = [1, 2, 3, 4, 5] if "0" in self.task_id else [6, 7, 8, 9, 10]

                data_msg: DataMessage = {
                    "data": data,
                    "task_id": self.task_id,
                    "timestamp": time.time()
                }

                data_channel.send(f"data:{self.task_id}", data_msg)

                # Update producer counter
                counter = data_channel.get("producer_count", 0)
                data_channel.set("producer_count", counter + 1)

                return f"produced: {data}"

            def consume_data(self):
                """Consumer task using typed channels."""
                data_channel = self.ctx.get_typed_channel(DataMessage)

                # Wait for producers to complete
                timeout = 5.0
                start_time = time.time()
                while time.time() - start_time < timeout:
                    producer_count = data_channel.get("producer_count", 0)
                    if producer_count >= 2:  # Wait for 2 producers
                        break
                    time.sleep(0.1)

                # Consume data from producers
                consumed_data = []
                for key in data_channel.keys():
                    if key.startswith("data:producer"):
                        data_msg = data_channel.get(key)
                        if data_msg and isinstance(data_msg, dict):
                            consumed_data.extend(data_msg["data"])

                result = f"consumed_{self.task_id}: {consumed_data}"

                # Send result using typed channel
                result_channel = self.ctx.get_typed_channel(TaskResultMessage)
                result_msg: TaskResultMessage = {
                    "task_id": self.task_id,
                    "result": consumed_data,
                    "timestamp": time.time(),
                    "status": "completed"
                }
                result_channel.set(f"consumed:{self.task_id}", result_msg)

                return result

            def aggregate_data(self):
                """Aggregator task using typed channels."""
                result_channel = self.ctx.get_typed_channel(TaskResultMessage)

                # Wait for consumers to complete
                timeout = 5.0
                start_time = time.time()
                aggregated_results = []

                while time.time() - start_time < timeout:
                    consumed_keys = [k for k in result_channel.keys() if k.startswith("consumed:")]
                    if len(consumed_keys) >= 2:  # Wait for 2 consumers
                        for key in consumed_keys:
                            result_data = result_channel.get(key)
                            if result_data and isinstance(result_data, dict):
                                aggregated_results.extend(result_data["result"])
                        break
                    time.sleep(0.1)

                # Create final result
                final_result = f"aggregated: {len(aggregated_results)} items total"

                final_msg: TaskResultMessage = {
                    "task_id": self.task_id,
                    "result": final_result,
                    "timestamp": time.time(),
                    "status": "completed"
                }
                result_channel.send("final_result", final_msg)

                return final_result

            def simple_task(self):
                """Simple task using channels."""
                channel = self.ctx.get_channel()
                channel.set(f"simple:{self.task_id}", f"executed_{self.task_id}")
                return f"completed_{self.task_id}"

        return ChannelTask(task_id, task_context)


@pytest.fixture
def task_handler(clean_redis):
    """Create task handler with Redis channel access."""
    return ChannelAwareTaskHandler(clean_redis)


@pytest.fixture
def task_worker(redis_queue, task_handler):
    """Create TaskWorker."""
    worker = TaskWorker(
        queue=redis_queue,
        handler=task_handler,
        worker_id="test_worker",
        max_concurrent_tasks=3,
        poll_interval=0.1
    )
    yield worker
    if worker.is_running:
        worker.stop()


def test_producer_consumer_with_typed_channels(redis_queue, task_worker, clean_redis):
    """Test producer-consumer pattern using typed channels inside tasks."""

    # Create producer tasks
    for i in range(2):
        task_spec = TaskSpec(
            task_id=f"producer_task_{i}",
            execution_context=redis_queue.execution_context,
            status=TaskStatus.READY
        )
        redis_queue.enqueue(task_spec)

    # Create consumer tasks
    for i in range(2):
        task_spec = TaskSpec(
            task_id=f"consumer_task_{i}",
            execution_context=redis_queue.execution_context,
            status=TaskStatus.READY
        )
        redis_queue.enqueue(task_spec)

    # Create aggregator task
    task_spec = TaskSpec(
        task_id="aggregator_task",
        execution_context=redis_queue.execution_context,
        status=TaskStatus.READY
    )
    redis_queue.enqueue(task_spec)

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
        handler = ChannelAwareTaskHandler(clean_redis)
        worker = TaskWorker(
            queue=redis_queue,
            handler=handler,
            worker_id=f"worker_{i}",
            max_concurrent_tasks=2,
            poll_interval=0.1
        )
        workers.append(worker)

    try:
        # Create coordination barrier
        barrier_id = "task_sync"
        redis_coordinator.create_barrier(barrier_id, 4)  # 4 parallel tasks

        # Create tasks
        for i in range(4):
            task_spec = TaskSpec(
                task_id=f"simple_task_{i}",
                execution_context=execution_context,
                status=TaskStatus.READY
            )
            redis_queue.enqueue(task_spec)

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
