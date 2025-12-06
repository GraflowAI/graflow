#!/usr/bin/env python3
"""
Test script for TaskExecutionContext implementation.
"""

from graflow.core.context import ExecutionContext, TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph
from graflow.core.task import Task


def test_task_execution_context_basic():
    """Test basic TaskExecutionContext functionality."""
    print("=== Testing TaskExecutionContext Basic Functionality ===")

    # Create execution context
    graph = TaskGraph()

    @task(inject_context=True)
    def test_task(task_ctx):
        print(f"Task ID: {task_ctx.task_id}")
        print(f"Session ID: {task_ctx.session_id}")
        print(f"Max cycles: {task_ctx.max_cycles}")
        print(f"Current cycle: {task_ctx.cycle_count}")
        print(f"Elapsed time: {task_ctx.elapsed_time():.4f}s")

        # Test local data
        task_ctx.set_local_data("test_key", "test_value")
        local_data = task_ctx.get_local_data("test_key")
        print(f"Local data: {local_data}")

        return f"Task executed: {task_ctx.task_id}"

    # Set up task
    graph.add_node(test_task, "test_task")
    exec_context = ExecutionContext.create(graph, max_steps=10, default_max_cycles=3)
    test_task.set_execution_context(exec_context)

    print("\n1. Testing direct task execution:")
    result = test_task()
    print(f"Result: {result}")

    print(f"Current task context stack size: {len(exec_context._task_execution_stack)}")


def test_context_manager():
    """Test context manager for task execution."""
    print("\n=== Testing Context Manager ===")

    graph = TaskGraph()
    exec_context = ExecutionContext.create(graph, max_steps=10, default_max_cycles=5)

    print("\n1. Testing context manager:")
    with exec_context.executing_task(Task("manual_task")) as task_ctx:
        print(f"Inside context: task_id = {task_ctx.task_id}")
        print(f"Can iterate: {task_ctx.can_iterate()}")
        print(f"Max cycles: {task_ctx.max_cycles}")

        # Test nested context
        with exec_context.executing_task(Task("nested_task")) as nested_ctx:
            print(f"Nested context: task_id = {nested_ctx.task_id}")
            print(f"Current task ID (should be nested): {exec_context.current_task_id}")

        print(f"Back to outer context: {exec_context.current_task_id}")

    print(f"Outside context: {exec_context.current_task_id}")


def test_cycle_management():
    """Test cycle management with TaskExecutionContext."""
    print("\n=== Testing Cycle Management ===")

    graph = TaskGraph()

    @task(inject_context=True)
    def cycle_task(task_ctx, data=None):
        if data is None:
            data = {"count": 0}

        count = data.get("count", 0)
        print(f"🔄 Cycle task - Count: {count}, Cycle: {task_ctx.cycle_count}")

        if count < 5 and task_ctx.can_iterate():
            print(f"  📊 Can iterate: cycles {task_ctx.cycle_count}/{task_ctx.max_cycles}")
            next_data = {"count": count + 1}
            iteration_id = task_ctx.next_iteration(next_data)
            print(f"  ➡️  Scheduled iteration: {iteration_id}")
        elif not task_ctx.can_iterate():
            print("  ⛔ Cannot iterate - cycle limit reached")
        else:
            print("  ✅ Count limit reached, stopping")

        return {"count": count, "cycle": task_ctx.cycle_count}

    # Set up task
    graph.add_node(cycle_task, "cycle_task")
    exec_context = ExecutionContext.create(graph, max_steps=20, default_max_cycles=3)
    cycle_task.set_execution_context(exec_context)

    print("\n1. Testing cycle management:")
    try:
        result = cycle_task()
        print(f"Final result: {result}")
    except ValueError as e:
        print(f"❌ Expected cycle limit error: {e}")


def test_parallel_context_simulation():
    """Test simulated parallel execution contexts."""
    print("\n=== Testing Parallel Context Simulation ===")

    graph = TaskGraph()

    @task(inject_context=True)
    def parallel_task_a(task_ctx: TaskExecutionContext):
        task_ctx.set_local_data("task_type", "A")
        print(f"🅰️  Task A executing - ID: {task_ctx.task_id}")
        print(f"    Local data: {task_ctx.get_local_data('task_type')}")
        return "Task A completed"

    @task(inject_context=True)
    def parallel_task_b(task_ctx: TaskExecutionContext):
        task_ctx.set_local_data("task_type", "B")
        print(f"🅱️  Task B executing - ID: {task_ctx.task_id}")
        print(f"    Local data: {task_ctx.get_local_data('task_type')}")
        return "Task B completed"

    @task(id="parallel_group_task", inject_context=True)
    def parallel_group_task(task_ctx: TaskExecutionContext):
        print(f"🔗 Parallel group executing - ID: {task_ctx.task_id}")

        # Simulate parallel execution by manually managing contexts
        results = []

        # Execute task A in its own context
        # Note: exec_context is captured from outer scope, but it will be defined by the time this runs
        with exec_context.executing_task(parallel_task_a) as ctx_a:
            result_a = parallel_task_a.func(ctx_a)
            results.append(result_a)

        # Execute task B in its own context
        with exec_context.executing_task(parallel_task_b) as ctx_b:
            result_b = parallel_task_b.func(ctx_b)
            results.append(result_b)

        print(f"  ✅ Parallel group completed: {results}")
        return results

    # Set up tasks
    # graph.add_node(parallel_group_task, "parallel_group_task")
    exec_context = ExecutionContext.create(graph, max_steps=20, default_max_cycles=5)
    parallel_group_task.set_execution_context(exec_context)

    print("\n1. Testing simulated parallel execution:")
    result = parallel_group_task()
    print(f"Final result: {result}")


def test_task_communication():
    """Test communication between tasks through TaskExecutionContext."""
    print("\n=== Testing Task Communication ===")

    graph = TaskGraph()

    @task(inject_context=True)
    def sender_task(task_ctx):
        print(f"📤 Sender task: {task_ctx.task_id}")
        channel = task_ctx.get_channel()
        channel.set("message_key", {"from": task_ctx.task_id, "data": "Hello from sender!"})
        print("  📨 Message sent to channel")
        return "Sender completed"

    @task(inject_context=True)
    def receiver_task(task_ctx):
        print(f"📥 Receiver task: {task_ctx.task_id}")
        channel = task_ctx.get_channel()
        message = channel.get("message_key")
        print(f"  📨 Received message: {message}")
        return f"Receiver got: {message}"

    # Set up tasks
    graph.add_node(sender_task, "sender_task")
    graph.add_node(receiver_task, "receiver_task")
    exec_context = ExecutionContext.create(graph, start_node="sender_task", max_steps=10)
    sender_task.set_execution_context(exec_context)
    receiver_task.set_execution_context(exec_context)

    print("\n1. Testing task communication:")
    sender_result = sender_task()
    receiver_result = receiver_task()
    print(f"Sender result: {sender_result}")
    print(f"Receiver result: {receiver_result}")


if __name__ == "__main__":
    try:
        test_task_execution_context_basic()
        test_context_manager()
        test_cycle_management()
        test_parallel_context_simulation()
        test_task_communication()
        print("\n🎉 All TaskExecutionContext tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
