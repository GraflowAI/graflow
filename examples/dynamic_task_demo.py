#!/usr/bin/env python3
"""
Dynamic Task Generation Demo

This example demonstrates the core dynamic task generation functionality
with next_task() and next_iteration() methods.
"""

import traceback

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.task import TaskWrapper
from graflow.core.workflow import workflow


@task(id="decision_task")
def make_decision(context: TaskExecutionContext, data):
    """Make a decision and create follow-up tasks dynamically."""
    print(f"🔍 Processing data: {data}")

    value = data.get("value", 0)

    if value > 10:  # noqa: PLR2004
        # Create high-value processing task
        high_task = TaskWrapper(
            "high_value_processor",
            lambda: process_high_value(value)
        )
        task_id = context.next_task(high_task)
        print(f"✨ Created high-value task: {task_id}")
    else:
        # Create low-value processing task
        low_task = TaskWrapper(
            "low_value_processor",
            lambda: process_low_value(value)
        )
        task_id = context.next_task(low_task)
        print(f"✨ Created low-value task: {task_id}")

    return {"decision_made": True, "value": value}


def process_high_value(value):
    """Process high-value data."""
    result = value * 2
    print(f"📈 High-value processing: {value} → {result}")
    return {"type": "high", "result": result}


def process_low_value(value):
    """Process low-value data."""
    result = value + 5
    print(f"📊 Low-value processing: {value} → {result}")
    return {"type": "low", "result": result}


@task(id="counter_task", inject_context=True)
def count_iterations(context: TaskExecutionContext, data=None):
    """Count iterations using next_iteration."""
    if data is None:
        data = {"count": 0, "limit": 3}
    count = data.get("count", 0)
    limit = data.get("limit", 3)

    print(f"🔢 Count: {count}/{limit}")

    if count < limit:
        # Continue iterating
        next_data = {"count": count + 1, "limit": limit}
        context.next_iteration(next_data)
        print(f"🔄 Scheduled next iteration: {count + 1}")
    else:
        # Done counting - create completion task
        done_task = TaskWrapper(
            "completion_task",
            lambda: complete_counting(count)
        )
        task_id = context.next_task(done_task)
        print(f"✅ Created completion task: {task_id}")

    return {"count": count, "completed": count >= limit}


def complete_counting(final_count):
    """Handle completion of counting."""
    print(f"🎉 Counting completed! Final count: {final_count}")
    return {"final_count": final_count, "status": "completed"}


def demo_conditional_tasks():
    """Demonstrate conditional dynamic task creation."""
    print("=" * 50)
    print("Demo 1: Conditional Dynamic Task Creation")
    print("=" * 50)

    # Test with high value (> 10)
    print("\n🚀 Testing with high value (15):")
    with workflow("high_value_demo") as wf:
        @task(id="init_high", inject_context=True)
        def init_high(context: TaskExecutionContext, _data=None):
            """Initialize high value processing."""
            return make_decision.func(context, {"value": 15})

        wf.execute("init_high")

    # Test with low value (<= 10)
    print("\n🚀 Testing with low value (7):")
    with workflow("low_value_demo") as wf:
        @task(id="init_low", inject_context=True)
        def init_low(context: TaskExecutionContext, _data=None):
            """Initialize low value processing."""
            return make_decision.func(context, {"value": 7})

        wf.execute("init_low")


def demo_iterative_tasks():
    """Demonstrate iterative processing with next_iteration."""
    print("\n" + "=" * 50)
    print("Demo 2: Iterative Processing")
    print("=" * 50)

    print("\n🚀 Starting iterative counting:")
    with workflow("iteration_demo") as wf:
        # Add the count_iterations task to the workflow
        wf.add_node("counter_task", count_iterations)

        wf.execute(start_node="counter_task")


def main():
    """Run all dynamic task generation demos."""
    print("🎯 Dynamic Task Generation Demo")
    print("This demo shows how to use context.next_task() and context.next_iteration()")

    try:
        demo_conditional_tasks()
        demo_iterative_tasks()

        print("\n" + "=" * 50)
        print("🎉 All demos completed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"❌ Error running demo: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
