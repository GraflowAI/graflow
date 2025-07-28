"""Advanced workflow features example."""

import threading
import time

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import WorkflowContext, get_current_workflow_context, workflow


def nested_workflows():
    """Demonstrate nested workflow contexts."""
    print("1. Nested Workflow Contexts:")
    with workflow("outer_workflow") as outer:
        @task
        def prepare_data():
            print("parepare_data() is called in outer workflow")

        @task(id="inner_workflow")
        def inner_workflow():
            # Nested context
            with workflow("inner_workflow") as inner:
                @task
                def process_data():
                    print("process_data() is called in inner workflow")

                @task
                def validate_data():
                    print("validate_data() is called in inner workflow")

                process_data >> validate_data
                #inner.show_info()

            print("Running inner workflow")
            inner.execute("process_data", max_steps=3)

        @task
        def finalize():
            print("finalize() is called in outer workflow")

        prepare_data >> inner_workflow >> finalize
        print("\nRun outer workflow:")
        #outer.show_info()
        outer.execute(max_steps=5)


def concurrent_workflow_execution():
    """Demonstrate concurrent workflow execution using threads."""
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


def workflow_composition_and_reuse():
    """Demonstrate workflow composition and reuse patterns."""
    print("3. Workflow Composition and Reuse:")

    def create_data_processing_workflow(name: str) -> WorkflowContext:
        """Factory function for creating reusable workflow patterns."""
        ctx = workflow(name)

        with ctx:
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

        return ctx

    # Create multiple instances of the same workflow pattern
    ctx1 = create_data_processing_workflow("ETL_Pipeline_1")
    ctx2 = create_data_processing_workflow("ETL_Pipeline_2")

    print("ETL Pipeline 1:")
    ctx1.show_info()
    print("\nETL Pipeline 2:")
    ctx2.show_info()

    print("\nExecuting both pipelines:")
    print("Pipeline 1:")
    ctx1.execute("load", max_steps=5)
    print("\nPipeline 2:")
    ctx2.execute("load", max_steps=5)


def global_fallback_behavior():
    """Demonstrate global fallback behavior for tasks outside workflow contexts."""
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
    global_context = get_current_workflow_context()
    print(f"Global pipeline structure: {global_context.show_info()}")

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

def dynamic_task_generation_with_loops():
    """Demonstrate dynamic task generation with loops."""
    print("5. Dynamic Task Generation with Loops:")

    with workflow("dynamic_generation") as ctx:

        @task(id="generate_tasks", inject_context=True)
        def generate_tasks(context: TaskExecutionContext):
            print("Generating tasks dynamically")
            for i in range(3):
                @task(id=f"dynamic_task_{i}")
                def dynamic_task(index=i):
                    print(f"Executing dynamic task {index}")

                # Register the task in the current context
                context.next_task(dynamic_task)

        @task
        def finalize():
            print("Finalizing dynamic task generation")

        generate_tasks >> finalize
        ctx.show_info()

        print("\nExecuting dynamic tasks:")
        ctx.execute("generate_tasks", max_steps=10)


def demonstrate_advanced_features():
    """Main function demonstrating all advanced workflow features."""
    print("=== Advanced Workflow Features ===\n")

    nested_workflows()
    print("\n" + "="*50 + "\n")

    concurrent_workflow_execution()
    print("\n" + "="*50 + "\n")

    workflow_composition_and_reuse()
    print("\n" + "="*50 + "\n")

    global_fallback_behavior()
    print("\n" + "="*50 + "\n")

    dynamic_task_generation_with_loops()


if __name__ == "__main__":
    demonstrate_advanced_features()
