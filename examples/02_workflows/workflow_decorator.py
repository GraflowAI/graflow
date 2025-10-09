"""
Workflow Decorator Example
===========================

This example demonstrates how to use the workflow() context manager to define
and execute coordinated task workflows in Graflow.

Concepts Covered:
-----------------
1. Creating a workflow context with workflow()
2. Defining tasks within the workflow scope
3. Setting up sequential task dependencies with >>
4. Executing workflows with ctx.execute()

Expected Output:
----------------
=== Data Pipeline Workflow ===

Starting execution from: extract_data
📥 Extracting data from source...
Data extracted: {'records': 1000, 'source': 'database'}

🔄 Transforming data...
Data transformed: 1000 records processed

💾 Loading data to destination...
Data loaded successfully

Execution completed after 3 steps

Pipeline completed successfully!
"""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Run a simple ETL (Extract-Transform-Load) workflow."""
    print("=== Data Pipeline Workflow ===\n")

    # Create a workflow context
    # All tasks defined within this context become part of the workflow
    with workflow("data_pipeline") as ctx:

        @task
        def extract_data():
            """Step 1: Extract data from a source."""
            print("📥 Extracting data from source...")
            data = {
                "records": 1000,
                "source": "database"
            }
            print(f"Data extracted: {data}\n")
            return data

        @task
        def transform_data():
            """Step 2: Transform the extracted data."""
            print("🔄 Transforming data...")
            # In a real workflow, this would receive data from extract_data
            # For this simple example, we just simulate the transformation
            result = "1000 records processed"
            print(f"Data transformed: {result}\n")
            return result

        @task
        def load_data():
            """Step 3: Load the transformed data."""
            print("💾 Loading data to destination...")
            # In a real workflow, this would receive data from transform_data
            # For this simple example, we just simulate the load
            print("Data loaded successfully\n")
            return "success"

        # Define the workflow execution order
        # >> operator means "then" (sequential execution)
        extract_data >> transform_data >> load_data # type: ignore

        # Execute the workflow starting from the first task
        # The workflow engine will execute tasks in dependency order:
        # 1. extract_data
        # 2. transform_data (after extract_data completes)
        # 3. load_data (after transform_data completes)
        ctx.execute("extract_data")

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **workflow() Context Manager**
#    - Creates a workflow execution environment
#    - Tasks defined within become part of the workflow
#    - Automatically manages task graph and execution
#
# 2. **Task Definition**
#    - Use @task decorator to define workflow tasks
#    - Tasks can return values (useful for data passing)
#    - Tasks execute in the order defined by dependencies
#
# 3. **Sequential Execution with >>**
#    - task1 >> task2 means "task2 runs after task1"
#    - Chain multiple tasks: task1 >> task2 >> task3
#    - Creates a directed acyclic graph (DAG) of dependencies
#
# 4. **Execution**
#    - ctx.execute(start_task) starts the workflow
#    - Engine executes tasks in topological order
#    - Each task completes before its successors start
#
# 5. **When to Use**
#    - Coordinated execution of multiple related tasks
#    - Sequential pipelines (ETL, data processing)
#    - When you need execution control and dependency management
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Add a validation step between extract and transform:
#    extract_data >> validate_data >> transform_data >> load_data
#
# 2. Add error handling to tasks (try/except)
#
# 3. Return and use actual data between tasks
#
# 4. Use max_steps parameter to limit execution:
#    ctx.execute("extract_data", max_steps=2)
#
# ============================================================================
