"""
Test script for WorkflowEngine integration with TaskExecutionContext.
"""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def test_engine_integration():
    """Test WorkflowEngine integration with TaskExecutionContext."""
    print("=== Testing WorkflowEngine Integration ===")

    with workflow("engine_integration_test") as wf:

        @task(inject_context=True)
        def task_a(task_ctx):
            print(f"🅰️  Task A - ID: {task_ctx.task_id}")
            print(f"    Session: {task_ctx.session_id}")
            print(f"    Elapsed: {task_ctx.elapsed_time():.4f}s")

            # Store some data
            task_ctx.set_local_data("processed_by", "task_a")

            # Use channel
            channel = task_ctx.get_channel()
            channel.set("task_a_result", {"status": "completed", "data": 42})

            return "Task A completed"

        @task(inject_context=True)
        def task_b(task_ctx):
            print(f"🅱️  Task B - ID: {task_ctx.task_id}")

            # Read from channel
            channel = task_ctx.get_channel()
            a_result = channel.get("task_a_result")
            print(f"    Task A result: {a_result}")

            # Test iteration capability
            if task_ctx.can_iterate():
                print(f"    Can iterate: {task_ctx.cycle_count}/{task_ctx.max_cycles}")

            return f"Task B got: {a_result}"

        # Set up graph
        _ = task_a >> task_b

        print("\n1. Testing engine execution:")
        result, exec_context = wf.execute(start_node="task_a", max_steps=10, ret_context=True)
        print("\nResults:")
        print(f"Task A result: {exec_context.get_result('task_a')}")
        print(f"Task B result: {exec_context.get_result('task_b')}")
        print(f"Final result: {result}")
        print(f"Executed tasks: {exec_context.completed_tasks}")
        print(f"Steps taken: {exec_context.steps}")


if __name__ == "__main__":
    try:
        test_engine_integration()
        print("\n🎉 Engine integration test passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
