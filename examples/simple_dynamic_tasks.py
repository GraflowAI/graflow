#!/usr/bin/env python3
"""
Simple Dynamic Task Generation Example

This example shows the basic usage of next_task() and next_iteration()
for dynamic workflow creation.
"""

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.task import TaskWrapper
from graflow.core.workflow import workflow


@task(id="decision_maker")
def make_decision(context: ExecutionContext, data):
    """Make a decision and create appropriate follow-up tasks."""
    print(f"Making decision based on: {data}")

    value = data.get("value", 0)

    if value > 10:
        # Create a "high value" processing task
        high_task = TaskWrapper(
            "process_high_value",
            lambda: print(f"Processing high value: {value}")
        )
        context.next_task(high_task)
        print("→ Created high value processing task")

    else:
        # Create a "low value" processing task
        low_task = TaskWrapper(
            "process_low_value",
            lambda: print(f"Processing low value: {value}")
        )
        context.next_task(low_task)
        print("→ Created low value processing task")

    return {"processed": True, "value": value}


@task(id="counter")
def count_up(context, data):
    """Count up and continue until reaching a limit."""
    current = data.get("count", 0)
    limit = data.get("limit", 5)

    print(f"Count: {current}")

    if current < limit:
        # Continue counting
        next_data = {"count": current + 1, "limit": limit}
        context.next_iteration(next_data)
        print(f"→ Scheduling next iteration with count {current + 1}")
    else:
        # Done counting, create completion task
        done_task = TaskWrapper(
            "counting_complete",
            lambda: print(f"Counting completed! Reached {current}")
        )
        context.next_task(done_task)
        print("→ Created completion task")

    return {"count": current, "done": current >= limit}


def main():
    """Run simple dynamic task examples."""
    print("Simple Dynamic Task Generation Examples")
    print("=" * 50)

    # Example 1: Conditional task creation
    print("\nExample 1: Conditional Task Creation")
    print("-" * 30)

    with workflow("conditional_tasks") as wf:
        # Test with high value - call the task function to initialize it
        make_decision(wf, {"value": 15})
        wf.execute()

    print("\n")
    with workflow("conditional_tasks_2") as wf:
        # Test with low value
        make_decision(wf, {"value": 5})
        wf.execute()

    # Example 2: Iterative processing
    print("\nExample 2: Iterative Processing")
    print("-" * 30)

    with workflow("iterative_counting") as wf:
        count_up(wf, {"count": 0, "limit": 3})
        wf.execute()

    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
