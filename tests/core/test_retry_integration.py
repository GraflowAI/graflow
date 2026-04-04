"""Integration tests for task retry via WorkflowEngine.

Tests verify the full retry lifecycle:
  task raises exception -> engine catches -> retry_controller checks budget
  -> task re-enqueued -> engine re-executes -> succeeds or exhausts retries.
"""

import pytest

from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.exceptions import GraflowRuntimeError, RetryLimitExceededError


def _run_workflow(
    graph: TaskGraph,
    start_node: str = "start",
    max_steps: int = 50,
    default_max_retries: int = 0,
) -> ExecutionContext:
    """Helper: create context, run engine, return context."""
    context = ExecutionContext.create(
        graph, start_node=start_node, max_steps=max_steps, default_max_retries=default_max_retries
    )
    engine = WorkflowEngine()
    engine.execute(context)
    return context


class TestRetryBasic:
    """Basic retry behavior tests."""

    def test_no_retry_by_default(self):
        """Without max_retries, a failing task raises immediately."""
        graph = TaskGraph()

        @task
        def failing():
            raise ValueError("boom")

        graph.add_node(failing, "failing")

        with pytest.raises(GraflowRuntimeError):
            _run_workflow(graph, start_node="failing")

    def test_retry_succeeds_after_failures(self):
        """Task retries and eventually succeeds."""
        graph = TaskGraph()
        attempt_count = 0

        @task
        def flaky():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"fail attempt {attempt_count}")
            return "success"

        graph.add_node(flaky, "flaky")
        context = _run_workflow(graph, start_node="flaky", default_max_retries=3)

        assert attempt_count == 3
        assert context.get_result("flaky") == "success"

    def test_retry_exhausted_raises_retry_limit_error(self):
        """Task that always fails exhausts retries and raises RetryLimitExceededError."""
        graph = TaskGraph()
        attempt_count = 0

        @task
        def always_fails():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("always fails")

        graph.add_node(always_fails, "always_fails")

        with pytest.raises(RetryLimitExceededError) as exc_info:
            _run_workflow(graph, start_node="always_fails", default_max_retries=2)

        # 1 initial + 2 retries = 3 attempts total
        assert attempt_count == 3
        assert exc_info.value.task_id == "always_fails"
        assert exc_info.value.max_retries == 2
        assert exc_info.value.last_error is not None

    def test_retry_count_tracked(self):
        """RetryController tracks the correct retry count."""
        graph = TaskGraph()
        attempt_count = 0

        @task
        def flaky():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("fail")
            return "ok"

        graph.add_node(flaky, "flaky")
        context = _run_workflow(graph, start_node="flaky", default_max_retries=3)

        assert context.retry_controller.get_retry_count("flaky") == 2


class TestRetryWithDecorator:
    """Tests for @task(max_retries=N) decorator parameter."""

    def test_decorator_max_retries_attribute(self):
        """@task(max_retries=N) stores the value on TaskWrapper."""

        @task(max_retries=3)
        def my_func():
            pass

        assert my_func.max_retries == 3

    def test_decorator_max_retries_default_none(self):
        """@task without max_retries defaults to None."""

        @task
        def my_func():
            pass

        assert my_func.max_retries is None

    def test_per_task_max_retries(self):
        """@task(max_retries=N) overrides global default."""
        graph = TaskGraph()
        attempt_count = 0

        @task(max_retries=2)
        def retryable():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("fail")
            return "recovered"

        graph.add_node(retryable, "retryable")
        # global default_max_retries=0, but per-task is 2
        context = _run_workflow(graph, start_node="retryable", default_max_retries=0)

        assert attempt_count == 3
        assert context.get_result("retryable") == "recovered"

    def test_per_task_max_retries_exhausted(self):
        """@task(max_retries=1) allows only 1 retry (2 total attempts)."""
        graph = TaskGraph()
        attempt_count = 0

        @task(max_retries=1)
        def limited():
            nonlocal attempt_count
            attempt_count += 1
            raise ValueError("always fails")

        graph.add_node(limited, "limited")

        with pytest.raises(RetryLimitExceededError) as exc_info:
            _run_workflow(graph, start_node="limited", default_max_retries=0)

        assert attempt_count == 2  # 1 initial + 1 retry
        assert exc_info.value.task_id == "limited"
        assert exc_info.value.max_retries == 1


class TestRetryWithContext:
    """Tests for retry state visible via TaskExecutionContext."""

    def test_retry_count_visible_in_context(self):
        """ctx.retry_count reflects the number of retries so far."""
        graph = TaskGraph()
        observed_counts: list[int] = []

        @task(inject_context=True, max_retries=3)
        def observable(ctx: TaskExecutionContext):
            observed_counts.append(ctx.retry_count)
            if len(observed_counts) < 3:
                raise ValueError("fail")
            return "ok"

        graph.add_node(observable, "observable")
        _run_workflow(graph, start_node="observable")

        # First attempt: retry_count=0, after 1st retry: 1, after 2nd retry: 2
        assert observed_counts == [0, 1, 2]

    def test_max_retries_visible_in_context(self):
        """ctx.max_retries reflects the configured max retries."""
        graph = TaskGraph()

        @task(inject_context=True, max_retries=5)
        def check_max(ctx: TaskExecutionContext):
            ctx.get_channel().set("max_retries", ctx.max_retries)
            return "ok"

        graph.add_node(check_max, "check_max")
        context = _run_workflow(graph, start_node="check_max")

        assert context.channel.get("max_retries") == 5


class TestRetryWithIteration:
    """Tests for retry behavior on iteration tasks."""

    def test_iteration_task_inherits_max_retries(self):
        """Iteration tasks created by next_iteration() inherit the base task's max_retries."""
        graph = TaskGraph()
        attempt_counts: dict[int, int] = {}  # cycle -> attempts

        @task(inject_context=True, max_retries=2, max_cycles=3)
        def retryable_iter(ctx: TaskExecutionContext, data=None):
            cycle = ctx.cycle_count
            attempt_counts[cycle] = attempt_counts.get(cycle, 0) + 1
            # Fail once on cycle 2
            if cycle == 2 and attempt_counts[cycle] == 1:
                raise ValueError("transient failure in iteration 2")
            ctx.get_channel().set("last_cycle", cycle)
            if ctx.can_iterate():
                ctx.next_iteration(data)

        graph.add_node(retryable_iter, "retryable_iter")
        context = _run_workflow(graph, start_node="retryable_iter")

        # All 3 cycles completed
        assert context.channel.get("last_cycle") == 3
        # Cycle 2 was attempted twice (1 fail + 1 retry success)
        assert attempt_counts[2] == 2

    def test_iteration_task_retry_exhausted(self):
        """Iteration task exhausting retries raises RetryLimitExceededError."""
        graph = TaskGraph()

        @task(inject_context=True, max_retries=1, max_cycles=5)
        def always_fails_iter(ctx: TaskExecutionContext, data=None):
            if ctx.cycle_count == 2:
                raise ValueError("always fails on cycle 2")
            if ctx.can_iterate():
                ctx.next_iteration(data)

        graph.add_node(always_fails_iter, "always_fails_iter")

        with pytest.raises(RetryLimitExceededError) as exc_info:
            _run_workflow(graph, start_node="always_fails_iter")

        assert exc_info.value.max_retries == 1


class TestRetryInPipeline:
    """Tests for retry behavior within multi-task pipelines."""

    def test_retry_does_not_affect_other_tasks(self):
        """A retrying task doesn't interfere with other tasks in the pipeline."""
        graph = TaskGraph()
        execution_log: list[str] = []
        flaky_attempts = 0

        @task
        def setup():
            execution_log.append("setup")
            return "initialized"

        @task(max_retries=2)
        def flaky_middle():
            nonlocal flaky_attempts
            flaky_attempts += 1
            execution_log.append(f"flaky:{flaky_attempts}")
            if flaky_attempts < 2:
                raise ValueError("transient error")
            return "recovered"

        @task
        def finalize():
            execution_log.append("finalize")
            return "done"

        graph.add_node(setup, "setup")
        graph.add_node(flaky_middle, "flaky_middle")
        graph.add_node(finalize, "finalize")
        graph.add_edge("setup", "flaky_middle")
        graph.add_edge("flaky_middle", "finalize")

        context = _run_workflow(graph, start_node="setup")

        assert "setup" in execution_log
        assert "flaky:1" in execution_log  # failed attempt
        assert "flaky:2" in execution_log  # successful retry
        assert "finalize" in execution_log
        assert context.get_result("flaky_middle") == "recovered"
