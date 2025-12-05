"""Example: Default strict mode for parallel group error handling.

This example demonstrates the default strict mode behavior where any task failure
in a parallel group causes the entire group to fail with a ParallelGroupError.

Key concepts:
- Default behavior requires all tasks to succeed
- ParallelGroupError contains details about failed and successful tasks
- Strict mode works with both DIRECT and THREADING backends
"""

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.engine import WorkflowEngine
from graflow.core.workflow import workflow
from graflow.exceptions import ParallelGroupError


def example_all_tasks_succeed():
    """Example: All tasks succeed - workflow completes successfully."""
    print("=" * 60)
    print("Example 1: All tasks succeed (strict mode)")
    print("=" * 60)

    with workflow("strict_mode_success") as wf:

        @task
        def validate_input():
            print("  [validate_input] Validating input data...")
            return {"status": "valid"}

        @task
        def fetch_data():
            print("  [fetch_data] Fetching data from API...")
            return {"records": 100}

        @task
        def check_permissions():
            print("  [check_permissions] Checking user permissions...")
            return {"authorized": True}

        # Create parallel group with threading backend (default strict mode)
        parallel = (validate_input | fetch_data | check_permissions).with_execution(
            backend=CoordinationBackend.THREADING
        )

        @task
        def process_results():
            print("  [process_results] Processing results...")
            return "success"

        parallel >> process_results

        # Execute workflow using engine
        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, parallel.task_id)
        engine.execute(context)

        assert context.get_result("process_results") == "success"

    print("\n✓ Workflow completed successfully")
    print()


def example_one_task_fails():
    """Example: One task fails - ParallelGroupError is raised."""
    print("=" * 60)
    print("Example 2: One task fails (strict mode)")
    print("=" * 60)

    with workflow("strict_mode_failure") as wf:

        @task
        def validate_input():
            print("  [validate_input] Validating input data...")
            return {"status": "valid"}

        @task
        def fetch_data():
            print("  [fetch_data] Fetching data from API...")
            raise ValueError("API connection timeout!")

        @task
        def check_permissions():
            print("  [check_permissions] Checking user permissions...")
            return {"authorized": True}

        # Create parallel group with default strict mode
        parallel = (validate_input | fetch_data | check_permissions).with_execution(
            backend=CoordinationBackend.THREADING
        )

        @task
        def process_results():
            print("  [process_results] Processing results...")
            return "success"

        parallel >> process_results

        # Execute workflow and handle error
        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, parallel.task_id)

        try:
            engine.execute(context)
        except ParallelGroupError as e:
            print(f"\n✗ Parallel group '{e.group_id}' failed!")
            print(f"  Failed tasks: {len(e.failed_tasks)}")
            for task_id, error_msg in e.failed_tasks:
                print(f"    - {task_id}: {error_msg}")
            print(f"  Successful tasks: {e.successful_tasks}")
            print("\n✓ Error handled gracefully")

            assert len(e.failed_tasks) == 1
            assert e.failed_tasks[0][0] == "fetch_data"
        else:
            raise AssertionError("Expected ParallelGroupError was not raised")

    print()


def example_multiple_failures():
    """Example: Multiple tasks fail - all failures are reported."""
    print("=" * 60)
    print("Example 3: Multiple tasks fail (strict mode)")
    print("=" * 60)

    with workflow("strict_mode_multiple_failures") as wf:

        @task
        def task_a():
            print("  [task_a] Executing...")
            raise RuntimeError("Task A failed!")

        @task
        def task_b():
            print("  [task_b] Executing...")
            return "success"

        @task
        def task_c():
            print("  [task_c] Executing...")
            raise ValueError("Task C failed!")

        # Create parallel group
        _parallel = (task_a | task_b | task_c).with_execution(
            backend=CoordinationBackend.THREADING
        )

        # Execute workflow and handle error
        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, _parallel.task_id)

        try:
            engine.execute(context)
        except ParallelGroupError as e:
            print(f"\n✗ Parallel group failed with {len(e.failed_tasks)} task failures:")
            for task_id, error_msg in e.failed_tasks:
                print(f"  - {task_id}: {error_msg}")
            print(f"\nSuccessful tasks: {e.successful_tasks}")
            print("\n✓ All failures captured and reported")

            assert len(e.failed_tasks) == 2
            assert set(e.successful_tasks) == {"task_b"}

    print()


def example_direct_backend():
    """Example: Strict mode works with DIRECT backend too."""
    print("=" * 60)
    print("Example 4: Strict mode with DIRECT backend")
    print("=" * 60)

    with workflow("strict_mode_direct") as wf:

        @task
        def task_a():
            print("  [task_a] Executing...")
            return "a"

        @task
        def task_b():
            print("  [task_b] Executing...")
            raise Exception("Task B failed!")

        # Use DIRECT backend (sequential execution, but same error handling)
        _parallel = (task_a | task_b).with_execution(backend=CoordinationBackend.DIRECT)

        engine = WorkflowEngine()
        context = ExecutionContext.create(wf.graph, _parallel.task_id)

        try:
            engine.execute(context)
        except ParallelGroupError as e:
            print("\n✗ Parallel group failed (DIRECT backend)")
            print(f"  Failed: {[tid for tid, _ in e.failed_tasks]}")
            print(f"  Successful: {e.successful_tasks}")
            print("\n✓ Strict mode works consistently across backends")

            assert [tid for tid, _ in e.failed_tasks] == ["task_b"]
        else:
            raise AssertionError("Expected ParallelGroupError was not raised")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PARALLEL GROUP ERROR HANDLING: STRICT MODE")
    print("=" * 60 + "\n")

    example_all_tasks_succeed()
    example_one_task_fails()
    example_multiple_failures()
    example_direct_backend()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
