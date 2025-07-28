#!/usr/bin/env python3
"""
Test script for context injection functionality.
"""

from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.graph import TaskGraph


def test_context_injection():
    """Test automatic context injection in task functions."""
    print("=== Testing Context Injection ===")

    # Create tasks with and without context injection
    @task
    def regular_task():
        return "Regular task executed"

    @task(inject_context=True)
    def context_task(context):
        print(f"Task ID: {context.current_task_id}")
        print(f"Session ID: {context.session_id}")
        print(f"Steps taken: {context.steps}")
        return f"Context task executed in session {context.session_id}"

    @task(inject_context=True, id="custom_context_task")
    def custom_context_task(context, message="default"):
        print(f"Custom task received: {message}")
        print(f"Current task ID: {context.current_task_id}")

        # Test next_task functionality
        @task
        def dynamic_task():
            return "Dynamic task created from context"

        task_id = context.next_task(dynamic_task)
        print(f"Created dynamic task: {task_id}")

        return f"Custom context task completed with message: {message}"

    # Create execution context
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "test_start", max_steps=10)

    # Set execution context for tasks
    regular_task.set_execution_context(context)
    context_task.set_execution_context(context)
    custom_context_task.set_execution_context(context)

    # Test regular task (no context injection)
    print("\n1. Testing regular task:")
    result1 = regular_task()
    print(f"Result: {result1}")

    # Test context injection
    print("\n2. Testing context injection:")
    result2 = context_task()
    print(f"Result: {result2}")

    # Test context injection with parameters
    print("\n3. Testing context injection with parameters:")
    result3 = custom_context_task("Hello from test!")
    print(f"Result: {result3}")

    # Test call vs run methods
    print("\n4. Testing run() method with context injection:")
    result4 = context_task.run()
    print(f"Run result: {result4}")

    print("\n=== Context Injection Test Complete ===")


def test_context_access_patterns():
    """Test different patterns for accessing context."""
    print("\n=== Testing Context Access Patterns ===")

    @task(inject_context=True)
    def analyze_context(context):
        print(f"Context type: {type(context)}")
        print(f"Available methods: {[m for m in dir(context) if not m.startswith('_')]}")

        # Test context operations
        channel = context.get_channel()
        print(f"Channel type: {type(channel)}")

        return "Context analysis complete"

    @task(inject_context=True)
    def iteration_example(context, data=None):
        if data is None:
            data = {"count": 0}

        count = data.get("count", 0)
        print(f"Iteration {count}")

        if count < 3:
            # Create next iteration
            next_data = {"count": count + 1}
            iteration_id = context.next_iteration(next_data, "iteration_example")
            print(f"Scheduled iteration: {iteration_id}")

        return f"Iteration {count} complete"

    # Setup
    graph = TaskGraph()
    context = ExecutionContext.create(graph, "analysis_start", max_steps=20)

    analyze_context.set_execution_context(context)
    iteration_example.set_execution_context(context)

    # Test context analysis
    print("\n1. Analyzing context:")
    context.current_task_id = "analyze_context"
    analyze_context()

    # Add iteration task to graph for testing
    graph.add_node("iteration_example", task=iteration_example)

    # Test iteration pattern
    print("\n2. Testing iteration pattern:")
    context.current_task_id = "iteration_example"
    iteration_example()

    print("\n=== Context Access Patterns Test Complete ===")


if __name__ == "__main__":
    try:
        test_context_injection()
        test_context_access_patterns()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()