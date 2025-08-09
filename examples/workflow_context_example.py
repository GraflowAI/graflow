"""Example of using WorkflowContext for scoped task management."""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    print("=== Context Manager Based Workflow Example ===\n")

    # Example 1: Simple workflow with context
    print("1. Simple Workflow Context:")
    with workflow("data_pipeline") as ctx:
        @task
        def extract_data():
            print("Extracting data from source")
            return {"raw_data": [1, 2, 3, 4, 5]}

        @task
        def transform_data():
            print("Transforming data")
            return {"processed_data": [2, 4, 6, 8, 10]}

        @task
        def load_data():
            print("Loading data to destination")
            return {"status": "loaded"}

        # Build pipeline
        extract_data >> transform_data >> load_data # type: ignore

        # Show workflow info
        ctx.show_info()
        print()

        # Execute workflow
        print("Executing data pipeline:")
        ctx.execute("extract_data", max_steps=5)

    print("\n" + "="*50 + "\n")

    # Example 2: Multiple parallel workflows
    print("2. Multiple Independent Workflows:")

    # First workflow - ML Training
    with workflow("ml_training") as ml_ctx:
        @task
        def load_dataset():
            print("Loading training dataset")

        @task
        def preprocess():
            print("Preprocessing data")

        @task
        def train_model():
            print("Training ML model")

        @task
        def evaluate():
            print("Evaluating model")

        # Build ML pipeline
        load_dataset >> preprocess >> train_model >> evaluate # type: ignore

        print("ML Training Workflow:")
        ml_ctx.show_info()

    # Second workflow - Data Analysis (independent)
    with workflow("data_analysis") as analysis_ctx:
        @task
        def collect_metrics():
            print("Collecting system metrics")

        @task
        def generate_report():
            print("Generating analysis report")

        @task
        def send_notification():
            print("Sending notification")

        # Build analysis pipeline
        collect_metrics >> generate_report >> send_notification # type: ignore

        print("\nData Analysis Workflow:")
        analysis_ctx.show_info()

    print("\n" + "="*50 + "\n")

    # Example 3: Complex workflow with parallel execution
    print("3. Complex Workflow with Parallel Tasks:")
    with workflow("complex_pipeline") as complex_ctx:
        @task
        def start_task():
            print("Starting complex pipeline")

        @task
        def parallel_task_a():
            print("Executing parallel task A")

        @task
        def parallel_task_b():
            print("Executing parallel task B")

        @task
        def parallel_task_c():
            print("Executing parallel task C")

        @task
        def merge_results():
            print("Merging results from parallel tasks")

        @task
        def final_task():
            print("Final processing")

        # Build complex pipeline with parallel execution
        start_task >> (parallel_task_a | parallel_task_b | parallel_task_c) >> merge_results >> final_task # type: ignore

        complex_ctx.show_info()
        complex_ctx.visualize_dependencies()
        print()

        print("Executing complex pipeline:")
        complex_ctx.execute("start_task", max_steps=10)

    print("\n" + "="*50 + "\n")

    # Example 4: Global task decorator with context fallback
    print("4. Global Task Decorator with Context Fallback:")

    # Global tasks (no context)
    @task
    def global_task_1():
        print("Global task 1")

    @task
    def global_task_2():
        print("Global task 2")

    # Context-aware tasks
    with workflow("mixed_context") as mixed_ctx:
        @task  # This will use the current context
        def context_task_1():
            print("Context task 1")

        @task  # This will also use the current context
        def context_task_2():
            print("Context task 2")

        # Build dependencies in context
        context_task_1 >> context_task_2 # type: ignore

        print("Mixed Context Workflow:")
        mixed_ctx.show_info()
        print()

        print("Executing mixed context workflow:")
        mixed_ctx.execute("context_task_1", max_steps=3)

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
