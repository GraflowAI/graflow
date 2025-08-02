"""Example using simplified workflow with existing classes."""

from graflow.core.decorators import task
from graflow.core.task import Task
from graflow.core.workflow import workflow


def main():  # noqa: PLR0915
    print("=== Simplified Workflow Example ===\n")

    # Example 1: Simple workflow with @task decorator
    print("1. Simple Workflow with @task:")
    with workflow("simple_pipeline") as ctx:
        @task
        def start():
            print("Starting pipeline")

        @task
        def process():
            print("Processing data")

        @task
        def finish():
            print("Finishing pipeline")

        # Build pipeline
        start >> process >> finish # type: ignore

        ctx.show_info()
        ctx.execute("start")

    print("\n" + "="*40 + "\n")

    # Example 2: Using Task class directly
    print("2. Using Task Class Directly:")
    with workflow("task_class_demo") as ctx:
        # Create tasks using Task class
        task_a = Task("task_a")
        task_b = Task("task_b")
        task_c = Task("task_c")

        # Build dependencies
        task_a >> task_b >> task_c # type: ignore

        ctx.show_info()
        ctx.execute("task_a")

    print("\n" + "="*40 + "\n")

    # Example 3: Mixed task types
    print("3. Mixed Task Types:")
    with workflow("mixed_demo") as ctx:
        # Task class
        manual_task = Task("manual_task")

        # Function-based task
        @task
        def decorated_task():
            print("Decorated task execution")

        @task
        def final_task():
            print("Final task execution")

        # Build pipeline
        manual_task >> decorated_task >> final_task # type: ignore

        ctx.show_info()
        ctx.execute("manual_task")

    print("\n" + "="*40 + "\n")

    # Example 4: Parallel execution
    print("4. Parallel Execution:")
    with workflow("parallel_demo") as ctx:
        @task
        def start_parallel():
            print("Starting parallel demo")

        @task
        def parallel_a():
            print("Parallel task A")

        @task
        def parallel_b():
            print("Parallel task B")

        @task
        def end_parallel():
            print("Ending parallel demo")

        # Create parallel pipeline
        start_parallel >> (parallel_a | parallel_b) >> end_parallel # type: ignore

        ctx.show_info()
        ctx.execute("start_parallel")

    print("\n" + "="*40 + "\n")

    # Example 5: Global tasks (no context)
    print("5. Global Tasks (No Context):")

    @task
    def global_task1():
        print("Global task 1")

    global_task2 = Task("global_task2")

    # These use global graph
    global_task1 >> global_task2 # type: ignore

    print("Global tasks created successfully")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
