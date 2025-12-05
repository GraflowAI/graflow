"""Test workflow termination and cancellation functionality."""

from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.exceptions import GraflowWorkflowCanceledError


def test_workflow_termination():
    """Test workflow termination (normal exit)."""
    executed_tasks = []

    with workflow("termination_test") as wf:
        @task(inject_context=True)
        def task_a(context):
            executed_tasks.append("task_a")
            # Terminate workflow early
            context.terminate_workflow("Early termination - condition met")
            return "a_result"

        @task
        def task_b():
            # Should not be executed
            executed_tasks.append("task_b")
            return "b_result"

        @task
        def task_c():
            # Should not be executed
            executed_tasks.append("task_c")
            return "c_result"

        _ = task_a >> task_b >> task_c

        # Create execution context manually to inspect completed_tasks
        from graflow.core.context import ExecutionContext
        from graflow.core.engine import WorkflowEngine
        exec_context = ExecutionContext.create(wf.graph, "task_a")
        engine = WorkflowEngine()
        engine.execute(exec_context)

    print("✓ Workflow terminated normally")
    print(f"  Executed tasks: {executed_tasks}")
    print("  Expected: ['task_a']")
    print(f"  Completed tasks: {exec_context.completed_tasks}")
    assert executed_tasks == ["task_a"], f"Expected ['task_a'], got {executed_tasks}"
    # task_a should be marked as completed (normal termination)
    assert "task_a" in exec_context.completed_tasks, "task_a should be marked as completed"
    print("✓ Test passed: task_a was executed and marked as completed")


def test_workflow_cancellation():
    """Test workflow cancellation (abnormal exit)."""
    executed_tasks = []
    exec_context = None

    try:
        with workflow("cancellation_test") as wf:
            @task(inject_context=True)
            def task_a(context):
                executed_tasks.append("task_a")
                # Cancel workflow
                context.cancel_workflow("Invalid data - canceling workflow")
                return "a_result"

            @task
            def task_b():
                # Should not be executed
                executed_tasks.append("task_b")
                return "b_result"

            @task
            def task_c():
                # Should not be executed
                executed_tasks.append("task_c")
                return "c_result"

            _ = task_a >> task_b >> task_c

            # Create execution context manually to inspect completed_tasks
            from graflow.core.context import ExecutionContext
            from graflow.core.engine import WorkflowEngine
            exec_context = ExecutionContext.create(wf.graph, "task_a")
            engine = WorkflowEngine()
            engine.execute(exec_context)

        print("✗ Expected GraflowWorkflowCanceledError but workflow completed")
        assert False, "Expected GraflowWorkflowCanceledError"
    except GraflowWorkflowCanceledError as e:
        print(f"✓ Workflow canceled with error: {e}")
        print(f"  Executed tasks: {executed_tasks}")
        print("  Expected: ['task_a']")
        print(f"  Completed tasks: {exec_context.completed_tasks}")
        assert executed_tasks == ["task_a"], f"Expected ['task_a'], got {executed_tasks}"
        assert e.task_id == "task_a", f"Expected task_id='task_a', got {e.task_id}"
        # task_a should NOT be marked as completed (abnormal cancellation)
        assert "task_a" not in exec_context.completed_tasks, "task_a should NOT be marked as completed"
        print("✓ Test passed: Workflow canceled, task_a NOT marked as completed")


def test_normal_workflow_without_control():
    """Test normal workflow execution without termination/cancellation."""
    executed_tasks = []

    with workflow("normal_test") as wf:
        @task
        def task_a():
            executed_tasks.append("task_a")
            return "a_result"

        @task
        def task_b():
            executed_tasks.append("task_b")
            return "b_result"

        @task
        def task_c():
            executed_tasks.append("task_c")
            return "c_result"

        task_a >> task_b >> task_c
        result = wf.execute("task_a")

    print("✓ Normal workflow completed")
    print(f"  Executed tasks: {executed_tasks}")
    print("  Expected: ['task_a', 'task_b', 'task_c']")
    assert executed_tasks == ["task_a", "task_b", "task_c"], f"Expected all tasks, got {executed_tasks}"
    print("✓ Test passed: All tasks executed")


def test_conditional_termination():
    """Test conditional workflow termination."""
    # Test with termination
    executed_tasks_1 = []

    with workflow("conditional_terminate_true") as wf:
        @task(inject_context=True)
        def check_condition_true(context):
            executed_tasks_1.append("check_condition")
            context.terminate_workflow("Condition met")
            return "checked"

        @task
        def task_b_1():
            executed_tasks_1.append("task_b")
            return "b_result"

        check_condition_true >> task_b_1
        wf.execute("check_condition_true")

    print("✓ Conditional termination (True) completed")
    print(f"  Executed tasks: {executed_tasks_1}")
    assert executed_tasks_1 == ["check_condition"], f"Expected ['check_condition'], got {executed_tasks_1}"

    # Test without termination
    executed_tasks_2 = []

    with workflow("conditional_terminate_false") as wf:
        @task(inject_context=True)
        def check_condition_false(context):
            executed_tasks_2.append("check_condition")
            # No termination
            return "checked"

        @task
        def task_b_2():
            executed_tasks_2.append("task_b")
            return "b_result"

        _ = check_condition_false >> task_b_2
        wf.execute("check_condition_false")

    print("✓ Conditional termination (False) completed")
    print(f"  Executed tasks: {executed_tasks_2}")
    assert executed_tasks_2 == ["check_condition", "task_b"], f"Expected both tasks, got {executed_tasks_2}"
    print("✓ Test passed: Conditional termination works correctly")


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Workflow Termination and Cancellation")
    print("=" * 80)

    print("\n1. Testing workflow termination (normal exit)...")
    test_workflow_termination()

    print("\n2. Testing workflow cancellation (abnormal exit)...")
    test_workflow_cancellation()

    print("\n3. Testing normal workflow without control...")
    test_normal_workflow_without_control()

    print("\n4. Testing conditional termination...")
    test_conditional_termination()

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
