#!/usr/bin/env python3
"""
Demonstration of how stack-based design supports parallel and nested execution.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph


def demo_stack_behavior():
    """Demonstrate the stack behavior during nested execution."""
    print("=== Stack Behavior Demonstration ===\n")

    graph = TaskGraph()
    exec_context = ExecutionContext.create(graph, "start", max_steps=20)

    def show_stack_state(label):
        stack = exec_context._task_execution_stack
        current = exec_context.current_task_id
        print(f"{label}:")
        print(f"  Stack size: {len(stack)}")
        print(f"  Current task: {current}")
        print(f"  Stack contents: {[ctx.task_id for ctx in stack]}")
        print()

    show_stack_state("1. Initial state")

    # Simulate nested execution
    with exec_context.executing_task("outer_task") as outer_ctx:
        show_stack_state("2. After entering outer_task")

        with exec_context.executing_task("middle_task") as middle_ctx:
            show_stack_state("3. After entering middle_task")

            with exec_context.executing_task("inner_task") as inner_ctx:
                show_stack_state("4. After entering inner_task (deepest)")

                # All contexts are available
                print(f"  Inner can access: task_id={inner_ctx.task_id}")
                print(f"  Current from stack: {exec_context.current_task_id}")
                print()

            show_stack_state("5. After exiting inner_task")

        show_stack_state("6. After exiting middle_task")

    show_stack_state("7. After exiting outer_task (back to empty)")


def demo_parallel_group_simulation():
    """Demonstrate how ParallelGroup would use the stack."""
    print("=== Parallel Group Simulation ===\n")

    graph = TaskGraph()
    exec_context = ExecutionContext.create(graph, "start", max_steps=20)

    @task(inject_context=True)
    def parallel_task_1(task_ctx):
        print(f"🅰️  Task 1 executing in context: {task_ctx.task_id}")
        task_ctx.set_local_data("task_number", 1)
        task_ctx.set_local_data("processing_time", 0.1)
        return f"Task 1 result: {task_ctx.get_local_data('task_number')}"

    @task(inject_context=True)
    def parallel_task_2(task_ctx):
        print(f"🅱️  Task 2 executing in context: {task_ctx.task_id}")
        task_ctx.set_local_data("task_number", 2)
        task_ctx.set_local_data("processing_time", 0.2)
        return f"Task 2 result: {task_ctx.get_local_data('task_number')}"

    @task(inject_context=True)
    def parallel_group(task_ctx):
        print(f"🔗 ParallelGroup executing in context: {task_ctx.task_id}")
        print(f"   Current stack depth: {len(exec_context._task_execution_stack)}")

        results = []

        # Each task gets its own isolated context
        print("\n  📋 Starting parallel task executions:")

        # Task 1 execution with its own context
        with exec_context.executing_task("parallel_task_1") as ctx1:
            print(f"    Stack during task 1: {[c.task_id for c in exec_context._task_execution_stack]}")
            result1 = parallel_task_1.func(ctx1)
            results.append(result1)
            print(f"    Task 1 local data: {ctx1.local_data}")

        # Task 2 execution with its own context
        with exec_context.executing_task("parallel_task_2") as ctx2:
            print(f"    Stack during task 2: {[c.task_id for c in exec_context._task_execution_stack]}")
            result2 = parallel_task_2.func(ctx2)
            results.append(result2)
            print(f"    Task 2 local data: {ctx2.local_data}")

        print(f"\n  ✅ ParallelGroup completed. Final stack: {[c.task_id for c in exec_context._task_execution_stack]}")
        return f"Parallel results: {results}"

    # Execute the parallel group
    print("1. Executing parallel group:")
    with exec_context.executing_task("parallel_group") as group_ctx:
        result = parallel_group.func(group_ctx)
        print(f"\nFinal result: {result}")


def demo_context_isolation():
    """Demonstrate how each task context is isolated."""
    print("\n=== Context Isolation Demonstration ===\n")

    graph = TaskGraph()
    exec_context = ExecutionContext.create(graph, "start", max_steps=20)

    # Simulate what happens in actual parallel execution
    print("1. Testing data isolation between contexts:")

    with exec_context.executing_task("task_a") as ctx_a:
        ctx_a.set_local_data("shared_key", "value_from_A")
        ctx_a.set_local_data("task_specific", "A_data")
        print(f"  Task A set data: {ctx_a.local_data}")

        # Nested task B (could be parallel in real scenario)
        with exec_context.executing_task("task_b") as ctx_b:
            ctx_b.set_local_data("shared_key", "value_from_B")  # Same key, different value
            ctx_b.set_local_data("task_specific", "B_data")
            print(f"  Task B set data: {ctx_b.local_data}")

            # Both contexts exist simultaneously
            print(f"  Current task: {exec_context.current_task_id}")
            print(f"  Stack: {[c.task_id for c in exec_context._task_execution_stack]}")

            # Each has its own isolated data
            print(f"  A's data unchanged: {ctx_a.local_data}")
            print(f"  B's data separate: {ctx_b.local_data}")

        print(f"  After B exits, A's data intact: {ctx_a.local_data}")


def demo_cycle_isolation():
    """Demonstrate how cycle counts are managed per task."""
    print("\n=== Cycle Count Isolation ===\n")

    graph = TaskGraph()
    exec_context = ExecutionContext.create(graph, "start", max_steps=20, default_max_cycles=3)

    print("1. Testing per-task cycle management:")

    # First task with its own cycle count
    with exec_context.executing_task("cycling_task_a") as ctx_a:
        print(f"  Task A initial cycles: {ctx_a.cycle_count}/{ctx_a.max_cycles}")
        ctx_a.register_cycle()
        ctx_a.register_cycle()
        print(f"  Task A after 2 cycles: {ctx_a.cycle_count}/{ctx_a.max_cycles}")

        # Second task with independent cycle count
        with exec_context.executing_task("cycling_task_b") as ctx_b:
            print(f"  Task B initial cycles: {ctx_b.cycle_count}/{ctx_b.max_cycles}")
            ctx_b.register_cycle()
            print(f"  Task B after 1 cycle: {ctx_b.cycle_count}/{ctx_b.max_cycles}")

            # Both tasks have independent cycle counts
            print(f"  Task A still has: {ctx_a.cycle_count} cycles")
            print(f"  Task B has: {ctx_b.cycle_count} cycles")
            print(f"  Tasks can iterate independently:")
            print(f"    A can iterate: {ctx_a.can_iterate()}")
            print(f"    B can iterate: {ctx_b.can_iterate()}")


if __name__ == "__main__":
    try:
        demo_stack_behavior()
        demo_parallel_group_simulation()
        demo_context_isolation()
        demo_cycle_isolation()
        print("\n🎉 All parallel support demonstrations completed!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()