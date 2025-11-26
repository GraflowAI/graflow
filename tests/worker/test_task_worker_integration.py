"""Tests for TaskWorker behavior."""

import time
from typing import Any, Optional

import pytest

from graflow.core.context import ExecutionContext
from graflow.core.graph import TaskGraph
from graflow.core.task import TaskWrapper
from graflow.queue.base import TaskQueue, TaskSpec
from graflow.queue.redis import RedisTaskQueue
from graflow.worker.worker import TaskWorker


def _create_execution_context() -> ExecutionContext:
    """Create ExecutionContext with isolated graph."""
    graph = TaskGraph()
    return ExecutionContext(graph)


def _create_registered_task(execution_context: ExecutionContext, task_id: str) -> TaskWrapper:
    """Create TaskWrapper registered with the execution context."""
    def task_fn():
        return f"result:{task_id}"

    wrapper = TaskWrapper(task_id, task_fn, register_to_context=False)
    unique_name = f"task_fn_{task_id}"
    wrapper.__name__ = unique_name
    wrapper.__qualname__ = unique_name
    wrapper.__module__ = __name__

    execution_context.graph.add_node(wrapper, task_id)
    # Task is stored in Graph, no need to register separately
    return wrapper


def _create_task_spec(execution_context: ExecutionContext, task_id: str) -> TaskSpec:
    """Create TaskSpec with registered executable."""
    executable = _create_registered_task(execution_context, task_id)
    return TaskSpec(executable=executable, execution_context=execution_context)


class MockTaskQueue(TaskQueue):
    """Simple in-memory queue for negative-path testing."""

    def __init__(self):
        super().__init__()
        self.tasks: list[TaskSpec] = []

    def enqueue(self, task_spec: TaskSpec) -> bool:
        self.tasks.append(task_spec)
        return True

    def dequeue(self) -> Optional[TaskSpec]:
        if self.tasks:
            return self.tasks.pop(0)
        return None

    def is_empty(self) -> bool:
        return not self.tasks

    def cleanup(self) -> None:
        self.tasks.clear()

    def to_list(self) -> list[str]:
        return [task.task_id for task in self.tasks]


def test_task_worker_rejects_non_redis_queue():
    """Ensure TaskWorker rejects queues that are not RedisTaskQueue."""
    queue = MockTaskQueue()

    with pytest.raises(ValueError, match="RedisTaskQueue"):
        TaskWorker(queue=queue, worker_id="invalid") # type: ignore


def test_task_worker_processes_tasks_with_redis_queue(clean_redis: Any):
    """Verify TaskWorker processes tasks correctly when using RedisTaskQueue."""
    execution_context = _create_execution_context()
    queue = RedisTaskQueue(redis_client=clean_redis, key_prefix="test_worker_queue")
    queue.cleanup()

    try:
        # Save initial graph snapshot for queue records
        execution_context.graph_hash = queue.graph_store.save(execution_context.graph)

        # Enqueue a few tasks
        for idx in range(3):
            queue.enqueue(_create_task_spec(execution_context, f"task_{idx}"))

        worker = TaskWorker(
            queue=queue,
            worker_id="redis_worker",
            max_concurrent_tasks=2,
            poll_interval=0.05
        )

        worker.start()

        # Wait until the queue is drained
        timeout = time.time() + 10.0
        while queue.size() > 0 and time.time() < timeout:
            time.sleep(0.1)

        worker.stop()

        metrics = worker.get_metrics()
        assert metrics["tasks_processed"] == 3
        assert metrics["tasks_failed"] == 0
        assert metrics["active_tasks"] == 0

    finally:
        queue.cleanup()
