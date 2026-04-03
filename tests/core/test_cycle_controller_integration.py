"""
Test script for cycle_controller integration in ExecutionContext.
"""

import pytest

from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph
from graflow.exceptions import CycleLimitExceededError

# Constants for test limits
COUNT_LIMIT = 5
ITERATION_LIMIT = 8
ATTEMPT_LIMIT = 5


def test_cycle_controller_integration():
    """Test cycle_controller integration with ExecutionContext.

    default_max_cycles=3: task executes 3 times (cycle_count 1,2,3).
    At cycle_count=3, accept_next_cycle() returns False so no more iterations.
    """
    graph = TaskGraph()
    observed_cycles: list[dict] = []

    @task(inject_context=True)
    def counting_task(ctx: TaskExecutionContext, data=None):
        if data is None:
            data = {"count": 0}

        count = data.get("count", 0)
        cycle_info = {"current": ctx.cycle_count, "max": ctx.max_cycles}
        observed_cycles.append(cycle_info)

        if count < COUNT_LIMIT and ctx.can_iterate():
            ctx.next_iteration({"count": count + 1})

        return {"count": count, "cycle_info": cycle_info}

    graph.add_node(counting_task, "counting_task")

    context = ExecutionContext.create(
        graph, start_node="counting_task", max_steps=20, default_max_cycles=3
    )
    engine = WorkflowEngine()
    engine.execute(context)

    # 3 executions: cycle_count 1, 2, 3
    assert len(observed_cycles) == 3
    assert observed_cycles[0] == {"current": 1, "max": 3}
    assert observed_cycles[1] == {"current": 2, "max": 3}
    assert observed_cycles[2] == {"current": 3, "max": 3}

    # Final cycle count stored in controller
    assert context.cycle_controller.get_cycle_count("counting_task") == 3


def test_custom_cycle_limits():
    """Test custom cycle limits for tasks.

    default_max_cycles=10, but per-node override is 5.
    Task iterates up to ITERATION_LIMIT=8 but is stopped at 5 by per-node limit.
    """
    graph = TaskGraph()
    observed_iterations: list[int] = []

    @task(inject_context=True)
    def limited_task(ctx: TaskExecutionContext, data=None):
        if data is None:
            data = {"iteration": 0}

        iteration = data.get("iteration", 0)
        observed_iterations.append(iteration)

        if iteration < ITERATION_LIMIT and ctx.can_iterate():
            ctx.next_iteration({"iteration": iteration + 1})

        return {"iteration": iteration}

    graph.add_node(limited_task, "limited_task")

    context = ExecutionContext.create(
        graph, start_node="limited_task", max_steps=30, default_max_cycles=10
    )
    # Set custom cycle limit (5 cycles)
    context.cycle_controller.set_node_max_cycles("limited_task", 5)

    engine = WorkflowEngine()
    engine.execute(context)

    # 5 executions: iteration 0,1,2,3,4; cycle_count 1..5
    assert len(observed_iterations) == 5
    assert observed_iterations == [0, 1, 2, 3, 4]
    assert context.cycle_controller.get_cycle_count("limited_task") == 5
    assert context.cycle_controller.get_max_cycles_for_node("limited_task") == 5


def test_cycle_info_methods():
    """Test cycle information methods via CycleController directly."""
    graph = TaskGraph()

    @task(inject_context=True)
    def info_task(_ctx: TaskExecutionContext):
        return "done"

    graph.add_node(info_task, "info_task")

    context = ExecutionContext.create(
        graph, start_node="info_task", max_steps=10, default_max_cycles=3
    )

    # Before any execution: cycle_count=0, accept_next_cycle=True
    ctrl = context.cycle_controller
    assert ctrl.get_cycle_count("info_task") == 0
    assert ctrl.accept_next_cycle("info_task") is True

    # Simulate manual increments
    ctrl.increment("info_task")  # count=1
    assert ctrl.get_cycle_count("info_task") == 1
    assert ctrl.accept_next_cycle("info_task") is True

    ctrl.increment("info_task")  # count=2
    assert ctrl.get_cycle_count("info_task") == 2
    assert ctrl.accept_next_cycle("info_task") is True

    # Hit the limit (max_cycles=3)
    ctrl.increment("info_task")  # count=3
    assert ctrl.get_cycle_count("info_task") == 3
    assert ctrl.accept_next_cycle("info_task") is False


def test_error_handling():
    """Test CycleLimitExceededError is raised when next_iteration() exceeds limit."""
    graph = TaskGraph()
    observed_attempts: list[int] = []

    @task(inject_context=True)
    def error_task(ctx: TaskExecutionContext, data=None):
        if data is None:
            data = {"attempt": 0}

        attempt = data.get("attempt", 0)
        observed_attempts.append(attempt)

        if attempt < ATTEMPT_LIMIT:
            ctx.next_iteration({"attempt": attempt + 1})

        return {"attempt": attempt}

    graph.add_node(error_task, "error_task")

    # default_max_cycles=2: task runs twice then next_iteration raises
    with pytest.raises(CycleLimitExceededError) as exc_info:
        context = ExecutionContext.create(
            graph, start_node="error_task", max_steps=10, default_max_cycles=2
        )
        engine = WorkflowEngine()
        engine.execute(context)

    assert exc_info.value.task_id == "error_task"
    assert exc_info.value.max_cycles == 2
    assert exc_info.value.cycle_count == 2
    # Executed twice (attempt 0 and 1), then next_iteration at cycle_count=2 raises
    assert observed_attempts == [0, 1]
