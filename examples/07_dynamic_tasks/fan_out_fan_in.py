"""
Fan-Out and Fan-Out-then-Fan-In Patterns with Dynamic Parallel Groups
======================================================================

This example demonstrates two patterns:

1. **Fan-Out Pattern**: Multiple branches each trigger their own integrator
2. **Fan-Out-then-Fan-In Pattern**: Branches diverge then converge to single integrator

Both use dynamic parallel groups and runtime task generation.

Prerequisites:
--------------
None

Concepts Covered:
-----------------
1. Dynamic parallel group creation with parallel()
2. Queueing parallel groups with context.next_task()
3. Fan-out: Each branch independently triggers subsequent tasks
4. Fan-out-then-fan-in: Branches diverge and then converge to single point
5. Dynamic path construction for state tracking
6. Runtime task generation (node1_1 dynamically creates node1_2)

Pattern 1: Fan-Out (integrator runs multiple times)
---------------------------------------------------

Task Graph:
           root
            |
      ┌─────┴─────┐
      |           |
    node1_1     node2
      |           |
    node1_2   integrator
      |
  integrator

Flow: root >> (node1_1 | node2) where each branch calls integrator independently
Result: Integrator runs TWICE (once per branch)

Expected Trace:
['root', 'root.node1_1', 'root.node1_1.node1_2', 'root.node1_1.node1_2.integrator',
 'root.node2', 'root.node2.integrator']

Pattern 2: Fan-Out and then Fan-In (integrator runs once)
-----------------------------------------

Task Graph:
           root
            |
      ┌─────┴─────┐
      |           |
    node1_1     node2
      |           |
    node1_2       |
      └─────┬─────┘
        integrator

Flow: root >> (node1_1 >> node1_2 | node2) >> integrator
Result: Integrator runs ONCE after both branches complete

Implementation: Uses >> operator to chain parallel group with integrator:
    parallel_group = parallel(node1_1, node2)
    chained = parallel_group >> integrator(parent_path=f"{current_path}.(node1_1.node1_2|node2)")
    context.next_task(chained)

Expected Trace:
['root', 'root.node1_1', 'root.node1_1.node1_2', 'root.node2', 'root.(node1_1.node1_2|node2).integrator']

Key Difference:
- Fan-Out: integrator appears multiple times (per branch)
- Fan-Out-then-Fan-In: integrator appears once at the end (after convergence)
"""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.task import parallel
from graflow.core.workflow import workflow


def main():
    """Fan-out and fan-out-then-fan-in patterns with dynamic parallel group creation."""
    print("=== Fan-Out Pattern ===\n")
    with workflow("fan_out") as ctx:

        @task(inject_context=True)
        def root(context: TaskExecutionContext, parent_path: str = "") -> None:
            """Root node that creates parallel group dynamically."""
            # Create and queue parallel group
            print("Root: Creating parallel group dynamically.")
            channel = context.get_channel()
            # Build current path
            current_path = "root" if not parent_path else f"{parent_path}.root"
            channel.append("trace", current_path)

            # Pass current_path as parent_path to children
            context.next_task(parallel(node1_1(parent_path=current_path), node2(parent_path=current_path)))

        @task(inject_context=True)
        def node1_1(context: TaskExecutionContext, parent_path: str = "") -> None:
            """Branch 1, step 1."""
            print("Node1_1: Executing branch 1, step 1.")

            channel = context.get_channel()
            current_path = f"{parent_path}.node1_1" if parent_path else "node1_1"
            channel.append("trace", current_path)

            context.next_task(node1_2(parent_path=current_path))

        @task(inject_context=True)
        def node1_2(context: TaskExecutionContext, parent_path: str = "") -> None:
            """Branch 1, step 2."""
            print("Node1_2: Executing branch 1, step 2.")

            channel = context.get_channel()
            current_path = f"{parent_path}.node1_2" if parent_path else "node1_2"
            channel.append("trace", current_path)

            # Jump to integrator
            context.next_task(integrator(parent_path=current_path))

        @task(inject_context=True)
        def node2(context: TaskExecutionContext, parent_path: str = "") -> None:
            """Branch 2."""
            print("Node2: Executing branch 2.")

            channel = context.get_channel()
            current_path = f"{parent_path}.node2" if parent_path else "node2"
            channel.append("trace", current_path)

            # Jump to integrator
            context.next_task(integrator(parent_path=current_path))

        @task(inject_context=True)
        def integrator(context: TaskExecutionContext, parent_path: str = "") -> str:
            """Integrator that combines all branch results."""
            print("Integrator: Combining branch results.")

            channel = context.get_channel()
            current_path = f"{parent_path}.integrator" if parent_path else "integrator"
            channel.append("trace", current_path)

            trace = channel.get("trace", [])
            combined = "|".join(trace)
            return combined

        # Execute workflow
        _, exec_ctx = ctx.execute("root", max_steps=20, ret_context=True)
        final_trace = exec_ctx.get_channel().get("trace", [])
        print("\nFinal Trace:", final_trace)

    print("\n" + "=" * 60)
    print("\n=== Fan-Out and then Fan-In Pattern (Proper Convergence) ===\n")

    with workflow("fan_out_fan_in") as ctx:

        @task(inject_context=True)
        def root(context: TaskExecutionContext, parent_path: str = "") -> None:
            """Root node that creates parallel group with fan-in using >> operator."""
            print("Root: Creating parallel group with fan-in.")
            channel = context.get_channel()
            current_path = "root" if not parent_path else f"{parent_path}.root"
            channel.append("trace", current_path)

            # Create parallel group and chain with integrator using >>
            # This ensures integrator runs once after both branches complete
            # Integrator's parent path explicitly shows it receives from both branches
            parallel_group = parallel(node1_1(parent_path=current_path), node2(parent_path=current_path))
            chained = parallel_group >> integrator(parent_path=f"{current_path}.(node1_1.node1_2|node2)")
            context.next_task(chained)

        @task(inject_context=True)
        def node1_1(context: TaskExecutionContext, parent_path: str = "") -> None:
            """Branch 1, step 1 - dynamically creates node1_2."""
            print("Node1_1: Executing branch 1, step 1.")
            channel = context.get_channel()
            current_path = f"{parent_path}.node1_1" if parent_path else "node1_1"
            channel.append("trace", current_path)

            # Dynamically create node1_2
            context.next_task(node1_2(parent_path=current_path))

        @task(inject_context=True)
        def node1_2(context: TaskExecutionContext, parent_path: str = "") -> None:
            """Branch 1, step 2."""
            print("Node1_2: Executing branch 1, step 2.")
            channel = context.get_channel()
            current_path = f"{parent_path}.node1_2" if parent_path else "node1_2"
            channel.append("trace", current_path)
            print("Node1_2: Branch 1 complete")

        @task(inject_context=True)
        def node2(context: TaskExecutionContext, parent_path: str = "") -> None:
            """Branch 2."""
            print("Node2: Executing branch 2.")
            channel = context.get_channel()
            current_path = f"{parent_path}.node2" if parent_path else "node2"
            channel.append("trace", current_path)
            print("Node2: Branch 2 complete")

        @task(inject_context=True)
        def integrator(context: TaskExecutionContext, parent_path: str = "") -> str:
            """Integrator that runs once after both branches complete."""
            print("Integrator: Combining results from both branches (runs only once).")
            channel = context.get_channel()
            current_path = f"{parent_path}.integrator" if parent_path else "integrator"
            channel.append("trace", current_path)

            trace = channel.get("trace", [])
            combined = "|".join(trace)
            return combined

        # Execute workflow
        _, exec_ctx = ctx.execute("root", max_steps=20, ret_context=True)
        final_trace = exec_ctx.get_channel().get("trace", [])
        print("\nFinal Trace:", final_trace)
        print("\n✓ Notice: Integrator appears only ONCE at the end")

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()


# ============================================================================
# Task Graph Visualizations:
# ============================================================================
#
# Fan-Out Pattern (integrator runs multiple times):
#
#           root
#            |
#      ┌─────┴─────┐
#      |           |
#    node1_1     node2
#      |           |
#    node1_2   integrator
#      |
#  integrator
#
# Each branch independently calls integrator, resulting in 2 integrator executions.
# Trace: ['root', 'root.node1_1', 'root.node1_1.node1_2',
#         'root.node1_1.node1_2.integrator', 'root.node2', 'root.node2.integrator']
#
# ----------------------------------------------------------------------------
#
# Fan-Out and then Fan-In Pattern (integrator runs once):
#
#           root
#            |
#      ┌─────┴─────┐
#      |           |
#    node1_1     node2
#      |           |
#    node1_2       |
#      └─────┬─────┘
#        integrator
#
# Both branches converge to single integrator using >> operator.
# Trace: ['root', 'root.node1_1', 'root.node1_1.node1_2',
#         'root.node2', 'root.(node1_1.node1_2|node2).integrator']
#
# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Fan-Out vs Fan-Out-then-Fan-In**
#
#    Fan-Out: Each branch independently triggers subsequent tasks
#      • node1_1 >> node1_2 >> integrator
#      • node2 >> integrator
#      • Result: integrator runs MULTIPLE times (once per branch)
#
#    Fan-Out-then-Fan-In: Branches diverge then converge to a single task
#      • root >> (node1_1 | node2) >> integrator
#      • First fans out to parallel branches, then fans in to single integrator
#      • Result: integrator runs ONCE after all branches complete
#
# 2. **Implementing Fan-In with >> Operator**
#
#    parallel_group = parallel(node1_1(...), node2(...))
#    chained = parallel_group >> integrator(parent_path=f"{current_path}.(node1_1.node1_2|node2)")
#    context.next_task(chained)
#
#    The >> operator automatically coordinates convergence.
#    Integrator's parent path explicitly shows it receives from both branches.
#    No manual counters, flags, or coordination logic needed!
#
# 3. **Dynamic Parallel Group Creation**
#
#    context.next_task(parallel(task1(parent_path=path), task2(parent_path=path)))
#
#    • Use parallel() to create ParallelGroup
#    • Pass bound parameters using task(param=value) syntax
#    • Queue the entire group with context.next_task()
#
# 4. **Runtime Task Generation**
#
#    Tasks can dynamically create and queue new tasks during execution:
#
#    @task(inject_context=True)
#    def node1_1(context: TaskExecutionContext, parent_path: str = "") -> None:
#        # Dynamically create node1_2
#        context.next_task(node1_2(parent_path=current_path))
#
#    This allows adaptive workflow structures based on runtime conditions.
#
# 5. **Path Construction with Bound Parameters**
#
#    • Pass parent_path as bound parameter: task(parent_path="root")
#    • Build path in task: current_path = f"{parent_path}.{task_name}"
#    • Store path in channel: channel.append("trace", current_path)
#
#    Benefits:
#    • Eliminates race conditions from shared state
#    • Each task knows its correct position in hierarchy
#    • Independent of execution order
#
# 6. **Key Patterns Demonstrated**
#
#    ✓ Dynamic parallel group creation at runtime
#    ✓ Fan-out: Independent branch execution with separate endpoints
#    ✓ Fan-out-then-fan-in: Converging branches to single endpoint using >> operator
#    ✓ Runtime task generation with next_task()
#    ✓ Hierarchical path tracking with bound parameters
#    ✓ No manual coordination logic required for convergence
#
# ============================================================================
