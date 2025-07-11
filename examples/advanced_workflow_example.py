"""Advanced workflow features example."""

import threading
import time

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def demonstrate_advanced_features():  # noqa: PLR0915
    print("=== Advanced Workflow Features ===\n")

    # 1. Nested workflows
    print("1. Nested Workflow Contexts:")
    with workflow("outer_workflow") as outer:
        @task
        def prepare_data():
            print("Preparing data in outer workflow")

        # Nested context
        with workflow("inner_workflow") as inner:
            @task
            def process_data():
                print("Processing data in inner workflow")

            @task
            def validate_data():
                print("Validating data in inner workflow")

            process_data >> validate_data
            inner.show_info()

        @task
        def finalize():
            print("Finalizing in outer workflow")

        prepare_data >> finalize
        print("\nOuter workflow:")
        outer.show_info()

    print("\n" + "="*50 + "\n")

    # 2. Concurrent workflows
    print("2. Concurrent Workflow Execution:")

    def worker_workflow(worker_id):
        with workflow(f"worker_{worker_id}") as ctx:
            @task
            def start_work():
                print(f"Worker {worker_id}: Starting work")
                time.sleep(0.1)  # Simulate work

            @task
            def do_processing():
                print(f"Worker {worker_id}: Processing")
                time.sleep(0.1)  # Simulate work

            @task
            def finish_work():
                print(f"Worker {worker_id}: Finishing work")

            start_work >> do_processing >> finish_work

            print(f"Worker {worker_id} workflow:")
            ctx.show_info()
            ctx.execute("start_work", max_steps=5)

    # Run multiple workflows concurrently
    threads = []
    for i in range(3):
        thread = threading.Thread(target=worker_workflow, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all workers to complete
    for thread in threads:
        thread.join()

    print("\n" + "="*50 + "\n")

    # 3. Workflow composition and reuse
    print("3. Workflow Composition and Reuse:")

    def create_data_processing_workflow(name):
        """Factory function for creating reusable workflow patterns."""
        ctx = workflow(name)

        @task
        def load():
            print(f"{name}: Loading data")

        @task
        def transform():
            print(f"{name}: Transforming data")

        @task
        def save():
            print(f"{name}: Saving data")

        # Setup pipeline
        load >> transform >> save

        return ctx, load, transform, save

    # Create multiple instances of the same workflow pattern
    ctx1, load1, transform1, save1 = create_data_processing_workflow("ETL_Pipeline_1")
    ctx2, load2, transform2, save2 = create_data_processing_workflow("ETL_Pipeline_2")

    print("ETL Pipeline 1:")
    ctx1.show_info()
    print("\nETL Pipeline 2:")
    ctx2.show_info()

    print("\nExecuting both pipelines:")
    print("Pipeline 1:")
    ctx1.execute("load", max_steps=5)
    print("\nPipeline 2:")
    ctx2.execute("load", max_steps=5)

    print("\n" + "="*50 + "\n")

    # 4. Global fallback behavior
    print("4. Global Fallback Behavior:")

    # These tasks will use global registration (no context)
    @task
    def global_task_a():
        print("Global task A")

    @task
    def global_task_b():
        print("Global task B")

    # Build global dependencies
    global_task_a >> global_task_b

    print("Global tasks created outside of context")

    # Show that context tasks are isolated
    with workflow("isolated_context") as isolated:
        @task  # This uses the context, not global
        def context_task_a():
            print("Context task A")

        @task
        def context_task_b():
            print("Context task B")

        context_task_a >> context_task_b

        print("Isolated context workflow:")
        isolated.show_info()

        print("\nExecuting isolated context:")
        isolated.execute("context_task_a", max_steps=3)


if __name__ == "__main__":
    demonstrate_advanced_features()
