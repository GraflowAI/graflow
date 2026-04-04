"""Tests for next_iteration() called from within tasks via WorkflowEngine execution.

These tests verify the full cycle lifecycle:
  task calls ctx.next_iteration() -> iteration task queued -> engine executes it
  -> cycle_count incremented -> CycleLimitExceededError when limit reached.
"""

import pytest

from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.exceptions import CycleLimitExceededError


def _run_workflow(
    graph: TaskGraph,
    start_node: str = "start",
    max_steps: int = 50,
    default_max_cycles: int = 100,
) -> ExecutionContext:
    """Helper: create context, run engine, return context."""
    context = ExecutionContext.create(
        graph, start_node=start_node, max_steps=max_steps, default_max_cycles=default_max_cycles
    )
    engine = WorkflowEngine()
    engine.execute(context)
    return context


class TestNextIteration:
    """Tests for next_iteration() invoked inside a running task."""

    def test_basic_iteration_counts_up(self):
        """next_iteration() re-executes the same task with incremented data."""
        graph = TaskGraph()

        @task(inject_context=True)
        def counter(ctx: TaskExecutionContext, data=None):
            count = (data or {}).get("count", 0)
            ctx.get_channel().set("last_count", count)
            if count < 3:
                ctx.next_iteration({"count": count + 1})
            return count

        graph.add_node(counter, "counter")
        context = _run_workflow(graph, start_node="counter", default_max_cycles=10)

        assert context.channel.get("last_count") == 3

    def test_cycle_count_reflects_iteration_number(self):
        """ctx.cycle_count inside each iteration should reflect how many times the task has executed (1-based)."""
        graph = TaskGraph()
        max_iterations = 5
        observed_counts: list[int] = []

        @task(inject_context=True, max_cycles=max_iterations)
        def repeater(ctx: TaskExecutionContext):
            observed_counts.append(ctx.cycle_count)
            if ctx.can_iterate():
                ctx.next_iteration()

        graph.add_node(repeater, "repeater")
        _run_workflow(graph, start_node="repeater", default_max_cycles=100)

        # cycle_count is 1-based: first execution sees 1, then 2, 3, ..., 5
        assert observed_counts == [1, 2, 3, 4, 5]

    def test_cycle_limit_stops_iteration(self):
        """CycleLimitExceededError raised when next_iteration() exceeds max_cycles.

        next_iteration() raises at cycle_count == max_cycles, so the return
        value on that iteration is never stored. Prior iterations do store results.
        """
        graph = TaskGraph()

        @task(inject_context=True)
        def infinite_loop(ctx: TaskExecutionContext, data=None):
            ctx.next_iteration(data)
            return "should stop"

        graph.add_node(infinite_loop, "infinite_loop")

        context = ExecutionContext.create(
            graph,
            start_node="infinite_loop",
            max_steps=50,
            default_max_cycles=3,
        )
        engine = WorkflowEngine()
        with pytest.raises(CycleLimitExceededError) as exc_info:
            engine.execute(context)

        assert exc_info.value.task_id == "infinite_loop"
        assert exc_info.value.max_cycles == 3
        # Iterations 1,2 succeed and store "should stop"; iteration 3 raises before return
        assert context.get_result("infinite_loop") == "should stop"

    def test_per_node_max_cycles(self):
        """Per-node max_cycles overrides global default."""
        graph = TaskGraph()

        @task(inject_context=True)
        def limited(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            ctx.get_channel().set("last_n", n)
            if n < 100:  # would iterate forever without limit
                ctx.next_iteration({"n": n + 1})
            return n

        graph.add_node(limited, "limited")

        context = ExecutionContext.create(graph, start_node="limited", max_steps=50, default_max_cycles=100)
        context.cycle_controller.set_node_max_cycles("limited", 4)

        engine = WorkflowEngine()
        with pytest.raises(CycleLimitExceededError) as exc_info:
            engine.execute(context)

        assert exc_info.value.max_cycles == 4
        # max_cycles=4: task executes 4 times (cycle_count 1..4), next_iteration at count=4 raises
        assert context.channel.get("last_n") == 3

    def test_iteration_with_pipeline(self):
        """next_iteration() works when the iterating task is part of a larger pipeline."""
        graph = TaskGraph()
        execution_log: list[str] = []

        @task
        def setup():
            execution_log.append("setup")
            return "initialized"

        @task(inject_context=True)
        def retry_task(ctx: TaskExecutionContext, data=None):
            attempt = (data or {}).get("attempt", 0)
            execution_log.append(f"retry:{attempt}")
            if attempt < 2:
                ctx.next_iteration({"attempt": attempt + 1})
            return f"attempt_{attempt}"

        @task
        def finalize():
            execution_log.append("finalize")
            return "done"

        graph.add_node(setup, "setup")
        graph.add_node(retry_task, "retry_task")
        graph.add_node(finalize, "finalize")
        graph.add_edge("setup", "retry_task")
        graph.add_edge("retry_task", "finalize")

        _run_workflow(graph, start_node="setup", default_max_cycles=10)

        assert "setup" in execution_log
        assert "retry:0" in execution_log
        assert "retry:1" in execution_log
        assert "retry:2" in execution_log
        assert "finalize" in execution_log

    def test_can_iterate_guards_last_iteration(self):
        """ctx.can_iterate() returns False when cycle limit is reached, allowing graceful exit."""
        graph = TaskGraph()

        @task(inject_context=True)
        def checker(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            ctx.get_channel().set(f"can_iterate_{n}", ctx.can_iterate())
            if ctx.can_iterate():
                ctx.next_iteration({"n": n + 1})
            return n

        graph.add_node(checker, "checker")
        # max_cycles=3: 3 executions total. cycle_count 1,2 → can_iterate()=True, 3 → False
        context = _run_workflow(graph, start_node="checker", default_max_cycles=3)

        assert context.channel.get("can_iterate_0") is True
        assert context.channel.get("can_iterate_1") is True
        # Iteration 2: can_iterate() is False (cycle_count=3 == max_cycles, last execution)
        assert context.channel.get("can_iterate_2") is False

    def test_conditional_iteration_stops_early(self):
        """Task can decide not to call next_iteration(), ending the loop normally."""
        graph = TaskGraph()

        @task(inject_context=True)
        def conditional(ctx: TaskExecutionContext, data=None):
            value = (data or {}).get("value", 10)
            ctx.get_channel().set("final_value", value)
            if value > 1:
                ctx.next_iteration({"value": value // 2})
            return value

        graph.add_node(conditional, "conditional")
        context = _run_workflow(graph, start_node="conditional", default_max_cycles=100)

        # 10 -> 5 -> 2 -> 1 (stops)
        assert context.channel.get("final_value") == 1

    def test_iteration_data_passed_correctly(self):
        """Data passed to next_iteration() is received by the next execution."""
        graph = TaskGraph()
        received_data: list[dict] = []

        @task(inject_context=True)
        def collector(ctx: TaskExecutionContext, data=None):
            data = data or {"items": []}
            received_data.append(data)
            items = data["items"]
            if len(items) < 3:
                ctx.next_iteration({"items": [*items, len(items)]})
            return data

        graph.add_node(collector, "collector")
        _run_workflow(graph, start_node="collector", default_max_cycles=10)

        assert received_data[0] == {"items": []}
        assert received_data[1] == {"items": [0]}
        assert received_data[2] == {"items": [0, 1]}
        assert received_data[3] == {"items": [0, 1, 2]}

    def test_task_decorator_max_cycles(self):
        """@task(max_cycles=N) limits iteration without manual cycle_controller setup."""
        graph = TaskGraph()

        @task(inject_context=True, max_cycles=4)
        def limited(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            ctx.get_channel().set("last_n", n)
            if n < 100:  # would iterate forever without limit
                ctx.next_iteration({"n": n + 1})
            return n

        graph.add_node(limited, "limited")

        # Use high default_max_cycles — the decorator's max_cycles=4 should take effect
        with pytest.raises(CycleLimitExceededError) as exc_info:
            _run_workflow(graph, start_node="limited", default_max_cycles=100)

        assert exc_info.value.max_cycles == 4
        assert exc_info.value.task_id == "limited"

    def test_task_decorator_max_cycles_graceful_exit(self):
        """@task(max_cycles=N) works with can_iterate() for graceful termination."""
        graph = TaskGraph()

        @task(inject_context=True, max_cycles=3)
        def bounded(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            ctx.get_channel().set("last_n", n)
            if ctx.can_iterate():
                ctx.next_iteration({"n": n + 1})
            return n

        graph.add_node(bounded, "bounded")
        context = _run_workflow(graph, start_node="bounded", default_max_cycles=100)

        # max_cycles=3: 3 executions total (cycle_count 1,2 → iterate, 3 → exit)
        assert context.channel.get("last_n") == 2

    def test_task_decorator_max_cycles_none_uses_default(self):
        """@task without max_cycles uses global default_max_cycles."""
        graph = TaskGraph()

        @task(inject_context=True)
        def unlimited(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            ctx.get_channel().set("last_n", n)
            if n < 100:
                ctx.next_iteration({"n": n + 1})
            return n

        graph.add_node(unlimited, "unlimited")

        with pytest.raises(CycleLimitExceededError) as exc_info:
            _run_workflow(graph, start_node="unlimited", default_max_cycles=5)

        # Should use global default of 5
        assert exc_info.value.max_cycles == 5


class TestTaskDecoratorMaxCycles:
    """Unit tests for @task(max_cycles=N) attribute propagation."""

    def test_max_cycles_set_on_task_wrapper(self):
        """@task(max_cycles=N) stores the value on the TaskWrapper instance."""

        @task(max_cycles=5)
        def my_func():
            pass

        assert my_func.max_cycles == 5

    def test_max_cycles_default_is_none(self):
        """@task without max_cycles defaults to None."""

        @task
        def my_func():
            pass

        assert my_func.max_cycles is None

    def test_max_cycles_with_inject_context(self):
        """@task(inject_context=True, max_cycles=10) sets both attributes."""

        @task(inject_context=True, max_cycles=10)
        def my_func(ctx):
            pass

        assert my_func.inject_context is True
        assert my_func.max_cycles == 10

    def test_max_cycles_with_string_id(self):
        """@task("custom_id", max_cycles=3) sets max_cycles with custom ID."""

        @task("custom_id", max_cycles=3)
        def my_func():
            pass

        assert my_func.task_id == "custom_id"
        assert my_func.max_cycles == 3

    def test_max_cycles_with_keyword_id(self):
        """@task(id="custom_id", max_cycles=7) sets max_cycles with keyword ID."""

        @task(id="my_id", max_cycles=7)
        def my_func():
            pass

        assert my_func.task_id == "my_id"
        assert my_func.max_cycles == 7

    def test_max_cycles_propagated_to_cycle_controller(self):
        """max_cycles is applied to CycleController when task is executed."""

        @task(inject_context=True, max_cycles=5)
        def limited(ctx: TaskExecutionContext):
            ctx.get_channel().set("observed_max", ctx.max_cycles)

        graph = TaskGraph()
        graph.add_node(limited, "limited")
        context = ExecutionContext.create(graph, start_node="limited", default_max_cycles=100)

        engine = WorkflowEngine()
        engine.execute(context)

        # CycleController should have the per-node max_cycles
        assert context.cycle_controller.get_max_cycles_for_node("limited") == 5
        # Task observed the per-node value via TaskExecutionContext
        assert context.channel.get("observed_max") == 5

    def test_max_cycles_iterates_until_limit(self):
        """@task(max_cycles=5) iterates exactly 5 times then raises CycleLimitExceededError."""
        observed_counts: list[int] = []

        @task(inject_context=True, max_cycles=5)
        def limited(ctx: TaskExecutionContext):
            observed_counts.append(ctx.cycle_count)
            ctx.next_iteration()

        graph = TaskGraph()
        graph.add_node(limited, "limited")

        with pytest.raises(CycleLimitExceededError) as exc_info:
            _run_workflow(graph, start_node="limited", default_max_cycles=100)

        assert exc_info.value.task_id == "limited"
        assert exc_info.value.max_cycles == 5
        # Executed 5 times (cycle_count 1..5); next_iteration() at count=5 raises
        assert observed_counts == [1, 2, 3, 4, 5]
        assert exc_info.value.cycle_count == 5

    def test_max_cycles_iterates_with_can_iterate_guard(self):
        """@task(max_cycles=5) with can_iterate() guard completes gracefully after 5 iterations."""
        observed_counts: list[int] = []

        @task(inject_context=True, max_cycles=5)
        def limited(ctx: TaskExecutionContext):
            observed_counts.append(ctx.cycle_count)
            ctx.get_channel().set("cycle_count", ctx.cycle_count)
            if ctx.can_iterate():
                ctx.next_iteration()

        graph = TaskGraph()
        graph.add_node(limited, "limited")
        context = _run_workflow(graph, start_node="limited", default_max_cycles=100)

        # Executed 5 times: cycle_count 1..4 (can_iterate=True) + 5 (can_iterate=False)
        assert observed_counts == [1, 2, 3, 4, 5]
        assert context.channel.get("cycle_count") == 5


class TestNextIterationEdgeCases:
    """Tests for edge cases and multi-node scenarios."""

    def test_multiple_nodes_independent_cycle_counts(self):
        """Each node tracks its own cycle count independently."""
        graph = TaskGraph()
        counts_a: list[int] = []
        counts_b: list[int] = []

        @task(inject_context=True)
        def node_a(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            counts_a.append(ctx.cycle_count)
            if n < 2:
                ctx.next_iteration({"n": n + 1})
            return n

        @task(inject_context=True)
        def node_b(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            counts_b.append(ctx.cycle_count)
            if n < 3:
                ctx.next_iteration({"n": n + 1})
            return n

        graph.add_node(node_a, "node_a")
        graph.add_node(node_b, "node_b")
        graph.add_edge("node_a", "node_b")

        _run_workflow(graph, start_node="node_a", default_max_cycles=10)

        # node_a executes 3 times (n=0,1,2), node_b executes 4 times (n=0,1,2,3)
        assert counts_a == [1, 2, 3]
        assert counts_b == [1, 2, 3, 4]

    def test_per_node_max_cycles_does_not_affect_other_nodes(self):
        """Per-node max_cycles on one node does not limit another node."""
        graph = TaskGraph()

        @task(inject_context=True, max_cycles=2)
        def limited_node(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            ctx.get_channel().set("limited_last", n)
            if ctx.can_iterate():
                ctx.next_iteration({"n": n + 1})
            return n

        @task(inject_context=True)
        def unlimited_node(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            ctx.get_channel().set("unlimited_last", n)
            if n < 5:
                ctx.next_iteration({"n": n + 1})
            return n

        graph.add_node(limited_node, "limited_node")
        graph.add_node(unlimited_node, "unlimited_node")
        graph.add_edge("limited_node", "unlimited_node")

        context = _run_workflow(graph, start_node="limited_node", default_max_cycles=100)

        # limited_node: max_cycles=2, so 2 executions (can_iterate guard stops at 2nd)
        assert context.channel.get("limited_last") == 1
        # unlimited_node: uses default_max_cycles=100, iterates 6 times (n=0..5)
        assert context.channel.get("unlimited_last") == 5

    def test_cycle_count_consistent_across_iterations(self):
        """cycle_count increments consistently across iterations within a single workflow run."""
        graph = TaskGraph()
        observed: list[dict] = []

        @task(inject_context=True)
        def tracker(ctx: TaskExecutionContext, data=None):
            n = (data or {}).get("n", 0)
            observed.append(
                {
                    "n": n,
                    "cycle_count": ctx.cycle_count,
                    "max_cycles": ctx.max_cycles,
                    "can_iterate": ctx.can_iterate(),
                }
            )
            if n < 4:
                ctx.next_iteration({"n": n + 1})
            return n

        graph.add_node(tracker, "tracker")
        _run_workflow(graph, start_node="tracker", default_max_cycles=10)

        assert len(observed) == 5
        for i, entry in enumerate(observed):
            assert entry["n"] == i
            assert entry["cycle_count"] == i + 1  # 1-based
            assert entry["max_cycles"] == 10
            # can_iterate is True for cycle_count < max_cycles
            assert entry["can_iterate"] == (i + 1 < 10)

    def test_next_iteration_with_none_data(self):
        """next_iteration() without data argument works correctly."""
        graph = TaskGraph()
        call_count = 0

        @task(inject_context=True)
        def no_data_task(ctx: TaskExecutionContext, data=None):
            nonlocal call_count
            call_count += 1
            ctx.get_channel().set("calls", call_count)
            if call_count < 3:
                ctx.next_iteration()  # No data argument
            return call_count

        graph.add_node(no_data_task, "no_data_task")
        context = _run_workflow(graph, start_node="no_data_task", default_max_cycles=10)

        assert context.channel.get("calls") == 3

    def test_default_max_cycles_1_allows_single_execution(self):
        """default_max_cycles=1 allows exactly one execution, next_iteration raises."""
        graph = TaskGraph()

        @task(inject_context=True)
        def single_shot(ctx: TaskExecutionContext, data=None):
            ctx.get_channel().set("executed", True)
            ctx.next_iteration()
            return "done"

        graph.add_node(single_shot, "single_shot")

        with pytest.raises(CycleLimitExceededError) as exc_info:
            _run_workflow(graph, start_node="single_shot", default_max_cycles=1)

        assert exc_info.value.max_cycles == 1
        assert exc_info.value.cycle_count == 1

    def test_can_iterate_false_at_max_cycles_1(self):
        """With max_cycles=1, can_iterate() returns False on the first (and only) execution."""
        graph = TaskGraph()

        @task(inject_context=True, max_cycles=1)
        def once(ctx: TaskExecutionContext, data=None):
            ctx.get_channel().set("can_iterate", ctx.can_iterate())
            ctx.get_channel().set("cycle_count", ctx.cycle_count)
            return "done"

        graph.add_node(once, "once")
        context = _run_workflow(graph, start_node="once", default_max_cycles=100)

        assert context.channel.get("can_iterate") is False
        assert context.channel.get("cycle_count") == 1

    def test_max_cycles_boundary_values(self):
        """Verify exact boundary: max_cycles=N allows N executions, N+1th raises."""
        for limit in [1, 2, 5]:
            graph = TaskGraph()
            observed_counts: list[int] = []

            @task(inject_context=True, max_cycles=limit)
            def bounded(ctx: TaskExecutionContext, data=None):
                observed_counts.append(ctx.cycle_count)
                ctx.next_iteration()

            graph.add_node(bounded, "bounded")

            with pytest.raises(CycleLimitExceededError) as exc_info:
                _run_workflow(graph, start_node="bounded", default_max_cycles=100)

            assert observed_counts == list(range(1, limit + 1))
            assert exc_info.value.max_cycles == limit
            assert exc_info.value.cycle_count == limit
