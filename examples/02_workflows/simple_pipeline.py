"""
Simple Pipeline Example
=======================

The absolute simplest example - a basic 3-task pipeline.
This is the "Hello World" of Graflow workflows.

Prerequisites:
--------------
None

Concepts Covered:
-----------------
1. Basic task definition with @task decorator
2. Sequential task dependencies with >> operator
3. Workflow context creation
4. Graph visualization
5. Workflow execution

Expected Output:
----------------
=== Simple Pipeline Demo ===

Workflow Information:
Name: simple_pipeline
Tasks: 3
Dependencies: 2

Executing pipeline...
Starting!
Middle!
End!

✅ Pipeline completed successfully!
"""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Run the simplest possible workflow."""
    print("=== Simple Pipeline Demo ===\n")

    with workflow("simple_pipeline") as ctx:

        @task
        def start():
            """First task in the pipeline."""
            print("Starting!")

        @task
        def middle():
            """Middle task in the pipeline."""
            print("Middle!")

        @task
        def end():
            """Final task in the pipeline."""
            print("End!")

        # Define sequential pipeline: start -> middle -> end
        start >> middle >> end  # type: ignore

        # Show workflow information
        print("Workflow Information:")
        print(f"Name: {ctx.name}")
        print(f"Graph:\n {ctx.graph}")

        # Execute the workflow
        print("Executing pipeline...")
        ctx.execute("start")

        print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Task Definition**
#    @task decorator converts a regular function into a workflow task
#    def my_task():
#        pass
#
# 2. **Sequential Dependencies**
#    The >> operator chains tasks together:
#    task1 >> task2 >> task3
#    - task2 runs after task1 completes
#    - task3 runs after task2 completes
#
# 3. **Workflow Context**
#    with workflow("name") as ctx:
#        # Define tasks and dependencies here
#        ctx.execute("start_task")
#
# 4. **Execution**
#    ctx.execute("task_name") starts execution from the specified task
#    The workflow follows the dependency graph automatically
#
# 5. **Next Steps**
#    ✅ Try modifying the print messages
#    ✅ Add a fourth task
#    ✅ See examples/01_basics/basic_task.py for more task features
#    ✅ See examples/01_basics/parallel_tasks.py for parallel execution
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Add a fourth task:
#    @task
#    def cleanup():
#        print("Cleaning up!")
#
#    start >> middle >> end >> cleanup
#
# 2. Add task logic:
#    @task
#    def calculate():
#        result = 2 + 2
#        print(f"Result: {result}")
#        return result
#
# 3. Change execution order:
#    # Still define: start >> middle >> end
#    # But execute from middle:
#    ctx.execute("middle")  # Only middle and end will run
#
# 4. View the graph:
#    # The graph object contains all task information
#    print(f"All tasks: {list(ctx.graph.tasks.keys())}")
#
# ============================================================================
