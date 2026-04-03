"""
Task Iterations (Cycle Control)
================================

Demonstrates iterative task execution using next_iteration(),
@task(max_cycles=N), ctx.can_iterate(), and ctx.cycle_count.

Concepts:
  - ctx.cycle_count: 1-based counter (1 on first execution)
  - ctx.can_iterate(): True if cycle budget remains
  - ctx.next_iteration(data): Queue another execution of the same task
  - @task(max_cycles=N): Allow exactly N executions per task

Run:
  PYTHONPATH=. uv run python examples/07_dynamic_tasks/task_iterations.py
"""

from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.graph import TaskGraph


def example_basic_iteration():
    """Iterate a fixed number of times using max_cycles."""
    print("=== Basic Iteration (max_cycles=5) ===")
    graph = TaskGraph()

    @task(inject_context=True, max_cycles=5)
    def counter(ctx: TaskExecutionContext):
        print(f"  cycle {ctx.cycle_count}/{ctx.max_cycles}")
        if ctx.can_iterate():
            ctx.next_iteration()

    graph.add_node(counter, "counter")
    ExecutionContext.create(graph, start_node="counter")
    WorkflowEngine().execute(ExecutionContext.create(graph, start_node="counter"))


def example_data_passing():
    """Pass data between iterations to accumulate state."""
    print("\n=== Data Passing Between Iterations ===")
    graph = TaskGraph()

    @task(inject_context=True, max_cycles=4)
    def accumulator(ctx: TaskExecutionContext, data=None):
        total = (data or {}).get("total", 0) + 10
        print(f"  cycle {ctx.cycle_count}: total={total}")
        ctx.get_channel().set("total", total)
        if ctx.can_iterate():
            ctx.next_iteration({"total": total})

    graph.add_node(accumulator, "accumulator")
    context = ExecutionContext.create(graph, start_node="accumulator")
    WorkflowEngine().execute(context)
    print(f"  Final total: {context.channel.get('total')}")


def example_early_exit():
    """Stop iterating early when a condition is met."""
    print("\n=== Early Exit on Convergence ===")
    graph = TaskGraph()

    @task(inject_context=True, max_cycles=20)
    def optimizer(ctx: TaskExecutionContext, data=None):
        loss = (data or {}).get("loss", 1.0) * 0.5
        print(f"  cycle {ctx.cycle_count}: loss={loss:.4f}")
        if loss < 0.05:
            print(f"  Converged at cycle {ctx.cycle_count}")
            return
        if ctx.can_iterate():
            ctx.next_iteration({"loss": loss})

    graph.add_node(optimizer, "optimizer")
    WorkflowEngine().execute(ExecutionContext.create(graph, start_node="optimizer"))


def example_pipeline_with_iterations():
    """Iterating task that collects data, then hands off to a downstream task."""
    print("\n=== Pipeline: Collect -> Summarize ===")
    graph = TaskGraph()

    @task(inject_context=True, max_cycles=3)
    def collect(ctx: TaskExecutionContext, data=None):
        items = list((data or {}).get("items", []))
        items.append(f"item_{ctx.cycle_count}")
        print(f"  [collect] cycle {ctx.cycle_count}: gathered {items[-1]}")
        ctx.get_channel().set("items", items)
        if ctx.can_iterate():
            ctx.next_iteration({"items": items})

    @task(inject_context=True)
    def summarize(ctx: TaskExecutionContext):
        items = ctx.get_channel().get("items")
        print(f"  [summarize] collected {len(items)} items: {items}")

    graph.add_node(collect, "collect")
    graph.add_node(summarize, "summarize")
    graph.add_edge("collect", "summarize")

    WorkflowEngine().execute(ExecutionContext.create(graph, start_node="collect"))


if __name__ == "__main__":
    example_basic_iteration()
    example_data_passing()
    example_early_exit()
    example_pipeline_with_iterations()
