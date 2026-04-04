"""
Task Retries
=============

Demonstrates automatic retry on task failure using
@task(max_retries=N), ctx.retry_count, and ctx.max_retries.

Concepts:
  - @task(max_retries=N): Allow up to N retry attempts after failure
  - @task(retry_policy=RetryPolicy(...)): Exponential backoff on retry
  - ctx.retry_count: Number of retries so far (0 on first attempt)
  - ctx.max_retries: Configured retry limit
  - default_max_retries: Global default (0 = no retries)
  - Retry is automatic — the engine re-enqueues the task on exception

Run:
  PYTHONPATH=. uv run python examples/07_dynamic_tasks/task_retries.py
"""

from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.core.retry import RetryPolicy
from graflow.exceptions import GraflowRuntimeError


def example_retry_succeeds():
    """Task fails twice, then succeeds on third attempt."""
    print("=== Retry Succeeds After Transient Failures ===")
    graph = TaskGraph()
    attempts = 0

    @task(inject_context=True, max_retries=3)
    def flaky_api(ctx: TaskExecutionContext):
        nonlocal attempts
        attempts += 1
        print(f"  attempt {attempts} (retry_count={ctx.retry_count})")
        if attempts < 3:
            raise ConnectionError(f"Connection refused (attempt {attempts})")
        return "ok"

    graph.add_node(flaky_api, "flaky_api")
    context = ExecutionContext.create(graph, start_node="flaky_api")
    WorkflowEngine().execute(context)
    print(f"  Result: {context.get_result('flaky_api')}")


def example_retry_exhausted():
    """Task fails on every attempt and exhausts retries."""
    print("\n=== Retry Exhausted ===")
    graph = TaskGraph()
    attempts = 0

    @task(max_retries=2)
    def always_fails():
        nonlocal attempts
        attempts += 1
        raise ValueError(f"fail #{attempts}")

    graph.add_node(always_fails, "always_fails")
    try:
        WorkflowEngine().execute(ExecutionContext.create(graph, start_node="always_fails"))
    except GraflowRuntimeError:
        print(f"  Failed after {attempts} attempts (1 initial + 2 retries)")


def example_retry_in_pipeline():
    """Retry a middle task without affecting the rest of the pipeline."""
    print("\n=== Retry in Pipeline ===")
    graph = TaskGraph()
    middle_attempts = 0

    @task
    def step_1():
        print("  [step_1] ok")
        return "data"

    @task(max_retries=2)
    def step_2():
        nonlocal middle_attempts
        middle_attempts += 1
        print(f"  [step_2] attempt {middle_attempts}")
        if middle_attempts < 2:
            raise RuntimeError("transient")
        return "processed"

    @task
    def step_3():
        print("  [step_3] ok")
        return "done"

    graph.add_node(step_1, "step_1")
    graph.add_node(step_2, "step_2")
    graph.add_node(step_3, "step_3")
    graph.add_edge("step_1", "step_2")
    graph.add_edge("step_2", "step_3")

    context = ExecutionContext.create(graph, start_node="step_1")
    WorkflowEngine().execute(context)
    print(f"  Pipeline result: {context.get_result('step_3')}")


def example_global_default():
    """Set a global default_max_retries for all tasks."""
    print("\n=== Global default_max_retries ===")
    graph = TaskGraph()
    attempts = 0

    @task
    def unstable():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError(f"fail #{attempts}")
        return "recovered"

    graph.add_node(unstable, "unstable")
    # All tasks get 3 retries by default
    context = ExecutionContext.create(graph, start_node="unstable", default_max_retries=3)
    WorkflowEngine().execute(context)
    print(f"  Recovered after {attempts} attempts, result: {context.get_result('unstable')}")


def example_exponential_backoff():
    """Retry with exponential backoff using RetryPolicy."""
    print("\n=== Exponential Backoff ===")
    graph = TaskGraph()
    attempts = 0

    @task(
        retry_policy=RetryPolicy(
            max_retries=3,
            initial_interval=0.1,  # short for demo
            backoff_factor=2.0,    # 0.1s → 0.2s → 0.4s
        ),
    )
    def flaky_service():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectionError(f"timeout (attempt {attempts})")
        return "success"

    graph.add_node(flaky_service, "flaky_service")
    context = ExecutionContext.create(graph, start_node="flaky_service")
    WorkflowEngine().execute(context)
    print(f"  Recovered after {attempts} attempts, result: {context.get_result('flaky_service')}")


if __name__ == "__main__":
    example_retry_succeeds()
    example_retry_exhausted()
    example_retry_in_pipeline()
    example_global_default()
    example_exponential_backoff()
