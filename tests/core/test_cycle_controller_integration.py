#!/usr/bin/env python3
"""
Test script for cycle_controller integration in ExecutionContext.
"""

from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph

# Constants for test limits
COUNT_LIMIT = 5
ITERATION_LIMIT = 8
ATTEMPT_LIMIT = 5


def test_cycle_controller_integration():
    """Test cycle_controller integration with ExecutionContext."""
    print("=== Testing Cycle Controller Integration ===")

    # Create execution context
    graph = TaskGraph()

    @task(inject_context=True)
    def counting_task(ctx: TaskExecutionContext, data=None):
        if data is None:
            data = {"count": 0}

        count = data.get("count", 0)
        print(f"📊 Count: {count}")

        # Get cycle info
        cycle_info = {"current": ctx.cycle_count, "max": ctx.max_cycles}
        print(f"🔄 Cycle info: {cycle_info}")

        if count < COUNT_LIMIT and ctx.can_iterate():
            # Create next iteration
            next_data = {"count": count + 1}
            iteration_id = ctx.next_iteration(next_data)
            print(f"➡️  Scheduled iteration: {iteration_id}")
        elif not ctx.can_iterate():
            print("⛔ Cannot iterate - cycle limit reached")
        else:
            print("✅ Count limit reached, stopping")

        return {"count": count, "cycle_info": cycle_info}

    # Set up task in graph
    graph.add_node(counting_task, "counting_task")

    # Create execution context
    context = ExecutionContext.create(graph, start_node="counting_task", max_steps=20, default_max_cycles=3)
    counting_task.set_execution_context(context)

    print("\n1. Testing default cycle limit (3):")
    try:
        counting_task()
    except ValueError as e:
        print(f"❌ Expected error: {e}")

    task_ctx = context._task_contexts.get("counting_task")
    final_count = task_ctx.cycle_count if task_ctx else 0
    print(f"Final cycle count: {final_count}")


def test_custom_cycle_limits():
    """Test custom cycle limits for tasks."""
    print("\n=== Testing Custom Cycle Limits ===")

    # Create execution context with higher default
    graph = TaskGraph()

    @task(inject_context=True)
    def limited_task(ctx: TaskExecutionContext, data=None):
        if data is None:
            data = {"iteration": 0}

        iteration = data.get("iteration", 0)
        print(f"🎯 Limited task iteration: {iteration}")

        cycle_info = {"current": ctx.cycle_count, "max": ctx.max_cycles}
        print(f"🔄 Cycle info: {cycle_info}")

        if iteration < ITERATION_LIMIT and ctx.can_iterate():
            next_data = {"iteration": iteration + 1}
            iteration_id = ctx.next_iteration(next_data)
            print(f"➡️  Next iteration: {iteration_id}")
        else:
            print(f"🛑 Stopping at iteration {iteration}")

        return {"iteration": iteration}

    # Set up task
    graph.add_node(limited_task, "limited_task")

    # Create execution context
    context = ExecutionContext.create(graph, start_node="limited_task", max_steps=30, default_max_cycles=10)

    # Set custom cycle limit for this task (5 cycles)
    context.cycle_controller.set_node_max_cycles("limited_task", 5)

    limited_task.set_execution_context(context)
    # Create and push task context instead of setting current_task_id
    task_ctx = context.create_task_context("limited_task")
    context.push_task_context(task_ctx)

    print("\n1. Testing custom cycle limit (5):")
    try:
        limited_task()
    except ValueError as e:
        print(f"❌ Expected error: {e}")

    task_ctx = context._task_contexts.get("limited_task")
    final_count = task_ctx.cycle_count if task_ctx else 0
    print(f"Final cycle count: {final_count}")


def test_cycle_info_methods():
    """Test cycle information methods."""
    print("\n=== Testing Cycle Info Methods ===")

    graph = TaskGraph()
    context = ExecutionContext.create(graph, "test_start", max_steps=10, default_max_cycles=3)

    # Test with no current task
    print("\n1. Testing with no current task:")
    current_ctx = context.current_task_context
    if current_ctx:
        info = {"current": current_ctx.cycle_count, "max": current_ctx.max_cycles}
        can_iterate = current_ctx.can_iterate()
    else:
        info = {"current": 0, "max": 0}
        can_iterate = False
    print(f"Cycle info: {info}")
    print(f"Can iterate: {can_iterate}")

    # Set up a task
    @task(inject_context=True)
    def info_task(_ctx: TaskExecutionContext):
        print("📋 Task executing")
        return "done"

    graph.add_node(info_task, "info_task")

    # Create execution context
    context = ExecutionContext.create(graph, start_node="info_task", max_steps=10, default_max_cycles=3)
    info_task.set_execution_context(context)
    # Create and push task context instead of setting current_task_id
    task_ctx = context.create_task_context("info_task")
    context.push_task_context(task_ctx)

    print("\n2. Testing with current task (no cycles yet):")
    current_ctx = context.current_task_context
    if current_ctx:
        info = {"current": current_ctx.cycle_count, "max": current_ctx.max_cycles}
        print(f"Cycle info: {info}")
        print(f"Can iterate: {current_ctx.can_iterate()}")
    else:
        print("No current task context available")

    # Simulate some cycles
    print("\n3. Testing after manual cycle registration:")
    current_ctx = context.current_task_context
    if current_ctx:
        current_ctx.register_cycle()
        current_ctx.register_cycle()

        info = {"current": current_ctx.cycle_count, "max": current_ctx.max_cycles}
        print(f"Cycle info: {info}")
        print(f"Can iterate: {current_ctx.can_iterate()}")

        # Try one more cycle to hit the limit
        current_ctx.register_cycle()
        info = {"current": current_ctx.cycle_count, "max": current_ctx.max_cycles}
        print(f"After 3rd cycle: {info}")
        print(f"Can iterate: {current_ctx.can_iterate()}")
    else:
        print("No current task context available")


def test_error_handling():
    """Test error handling in cycle operations."""
    print("\n=== Testing Error Handling ===")

    graph = TaskGraph()

    @task(inject_context=True)
    def error_task(ctx: TaskExecutionContext, data=None):
        if data is None:
            data = {"attempt": 0}

        attempt = data.get("attempt", 0)
        print(f"💥 Error task attempt: {attempt}")

        try:
            if attempt < ATTEMPT_LIMIT:  # Try more than cycle limit
                next_data = {"attempt": attempt + 1}
                iteration_id = ctx.next_iteration(next_data)
                print(f"➡️  Next attempt: {iteration_id}")
            else:
                print("✅ Completed successfully")
        except ValueError as e:
            print(f"❌ Caught expected error: {e}")
            return {"error": str(e), "attempt": attempt}

        return {"attempt": attempt}

    # Set up task
    graph.add_node(error_task, "error_task")

    # Create execution context
    context = ExecutionContext.create(graph, start_node="error_task", max_steps=10, default_max_cycles=2)
    error_task.set_execution_context(context)
    # Create and push task context instead of setting current_task_id
    task_ctx = context.create_task_context("error_task")
    context.push_task_context(task_ctx)

    print("\n1. Testing cycle limit error:")
    result = error_task()
    print(f"Final result: {result}")


if __name__ == "__main__":
    try:
        test_cycle_controller_integration()
        test_custom_cycle_limits()
        test_cycle_info_methods()
        test_error_handling()
        print("\n🎉 All cycle controller tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
