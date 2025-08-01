"""Simple example showing context-aware @task decorator."""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():  # noqa: PLR0915
    print("=== Simple Context-Aware @task Example ===\n")

    # Example 1: Simple pipeline with automatic context detection
    print("1. Simple Pipeline:")
    with workflow("data_pipeline") as ctx:
        @task
        def extract():
            print("Extracting data")
            return "raw_data"

        @task
        def transform():
            print("Transforming data")
            return "clean_data"

        @task
        def load():
            print("Loading data")
            return "loaded"

        # Build pipeline - just use @task, no @ctx.task needed!
        extract >> transform >> load

        ctx.show_info()
        ctx.execute("extract")

    print("\n" + "="*40 + "\n")

    # Example 2: Multiple independent workflows
    print("2. Multiple Independent Workflows:")

    # First workflow
    with workflow("ml_pipeline") as ml:
        @task
        def load_data():
            print("ML: Loading dataset")

        @task
        def train():
            print("ML: Training model")

        load_data >> train

        print("ML Pipeline:")
        ml.show_info()

    # Second workflow (completely independent)
    with workflow("analytics") as analytics:
        @task
        def collect():
            print("Analytics: Collecting metrics")

        @task
        def report():
            print("Analytics: Generating report")

        collect >> report

        print("\nAnalytics Pipeline:")
        analytics.show_info()

    print("\n" + "="*40 + "\n")

    # Example 3: Parallel tasks
    print("3. Parallel Execution:")
    with workflow("parallel_demo") as parallel:
        @task
        def start():
            print("Starting parallel demo")

        @task
        def task_a():
            print("Executing task A")

        @task
        def task_b():
            print("Executing task B")

        @task
        def task_c():
            print("Executing task C")

        @task
        def finish():
            print("Finishing parallel demo")

        # Create parallel execution: start >> (A | B | C) >> finish
        start >> (task_a | task_b | task_c) >> finish

        parallel.show_info()
        parallel.execute("start")

    print("\n" + "="*40 + "\n")

    # Example 4: Global tasks (no context)
    print("4. Global Tasks (No Context):")

    @task
    def global_task1():
        print("Global task 1")

    @task
    def global_task2():
        print("Global task 2")

    # These will use the global graph
    global_task1 >> global_task2

    print("Global tasks created (will use global graph)")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
