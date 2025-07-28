#!/usr/bin/env python3
"""
Test script for cycle_controller integration in ExecutionContext.
"""

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph


def test_cycle_controller_integration():
    """Test cycle_controller integration with ExecutionContext."""
    print("=== Testing Cycle Controller Integration ===")

    # Create execution context
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "test_start", max_steps=20, default_max_cycles=3)

    @task(inject_context=True)
    def counting_task(ctx, data=None):
        if data is None:
            data = {"count": 0}

        count = data.get("count", 0)
        print(f"📊 Count: {count}")

        # Get cycle info
        cycle_info = ctx.get_cycle_info()
        print(f"🔄 Cycle info: {cycle_info}")

        if count < 5 and ctx.can_task_iterate():
            # Create next iteration
            next_data = {"count": count + 1}
            iteration_id = ctx.next_iteration(next_data)
            print(f"➡️  Scheduled iteration: {iteration_id}")
        elif not ctx.can_task_iterate():
            print("⛔ Cannot iterate - cycle limit reached")
        else:
            print("✅ Count limit reached, stopping")

        return {"count": count, "cycle_info": cycle_info}

    # Set up task in graph
    graph.add_node("counting_task", task=counting_task)
    counting_task.set_execution_context(context)

    print("\n1. Testing default cycle limit (3):")
    try:
        counting_task()
    except ValueError as e:
        print(f"❌ Expected error: {e}")

    print(f"Final cycle count: {context.get_cycle_count('counting_task')}")


def test_custom_cycle_limits():
    """Test custom cycle limits for tasks."""
    print("\n=== Testing Custom Cycle Limits ===")

    # Create execution context with higher default
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "test_start", max_steps=30, default_max_cycles=10)

    @task(inject_context=True)
    def limited_task(ctx, data=None):
        if data is None:
            data = {"iteration": 0}

        iteration = data.get("iteration", 0)
        print(f"🎯 Limited task iteration: {iteration}")

        cycle_info = ctx.get_cycle_info()
        print(f"🔄 Cycle info: {cycle_info}")

        if iteration < 8 and ctx.can_task_iterate():
            next_data = {"iteration": iteration + 1}
            iteration_id = ctx.next_iteration(next_data)
            print(f"➡️  Next iteration: {iteration_id}")
        else:
            print(f"🛑 Stopping at iteration {iteration}")

        return {"iteration": iteration}

    # Set custom cycle limit for this task (5 cycles)
    context.set_max_cycles_for_task("limited_task", 5)

    # Set up task
    graph.add_node("limited_task", task=limited_task)
    limited_task.set_execution_context(context)
    context.current_task_id = "limited_task"

    print("\n1. Testing custom cycle limit (5):")
    try:
        limited_task()
    except ValueError as e:
        print(f"❌ Expected error: {e}")

    print(f"Final cycle count: {context.get_cycle_count('limited_task')}")


def test_cycle_info_methods():
    """Test cycle information methods."""
    print("\n=== Testing Cycle Info Methods ===")

    graph = TaskGraph()
    context = ExecutionContext.create(graph, "test_start", max_steps=10, default_max_cycles=3)

    # Test with no current task
    print("\n1. Testing with no current task:")
    info = context.get_cycle_info()
    print(f"Cycle info: {info}")
    print(f"Can iterate: {context.can_task_iterate()}")

    # Set up a task
    @task(inject_context=True)
    def info_task(ctx):
        print("📋 Task executing")
        return "done"

    graph.add_node("info_task", task=info_task)
    info_task.set_execution_context(context)
    context.current_task_id = "info_task"

    print("\n2. Testing with current task (no cycles yet):")
    info = context.get_cycle_info()
    print(f"Cycle info: {info}")
    print(f"Can iterate: {context.can_task_iterate()}")

    # Simulate some cycles
    print("\n3. Testing after manual cycle registration:")
    context.cycle_controller.register_cycle("info_task")
    context.cycle_controller.register_cycle("info_task")

    info = context.get_cycle_info()
    print(f"Cycle info: {info}")
    print(f"Can iterate: {context.can_task_iterate()}")

    # Try one more cycle to hit the limit
    context.cycle_controller.register_cycle("info_task")
    info = context.get_cycle_info()
    print(f"After 3rd cycle: {info}")
    print(f"Can iterate: {context.can_task_iterate()}")


def test_error_handling():
    """Test error handling in cycle operations."""
    print("\n=== Testing Error Handling ===")

    graph = TaskGraph()
    context = ExecutionContext.create(graph, "test_start", max_steps=10, default_max_cycles=2)

    @task(inject_context=True)
    def error_task(ctx, data=None):
        if data is None:
            data = {"attempt": 0}

        attempt = data.get("attempt", 0)
        print(f"💥 Error task attempt: {attempt}")

        try:
            if attempt < 5:  # Try more than cycle limit
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
    graph.add_node("error_task", task=error_task)
    error_task.set_execution_context(context)
    context.current_task_id = "error_task"

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