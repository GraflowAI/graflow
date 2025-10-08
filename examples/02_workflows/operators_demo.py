"""
Workflow Operators Demo
========================

This example demonstrates the task composition operators in Graflow:
- >> (Sequential execution)
- |  (Parallel execution)

These operators allow you to build complex DAG (Directed Acyclic Graph) workflows
with sequential and parallel execution patterns.

Concepts Covered:
-----------------
1. Sequential execution with >> operator
2. Parallel execution with | operator
3. Combined patterns: (parallel) >> sequential
4. Building complex workflow DAGs

Expected Output:
----------------
=== Example 1: Sequential Execution ===
Starting execution from: step_1
▶️  Step 1: Fetch data
▶️  Step 2: Validate data
▶️  Step 3: Process data
Execution completed after 3 steps

=== Example 2: Parallel Execution ===
Starting execution from: task_a
⚡ Task A: Process partition 1
⚡ Task B: Process partition 2
⚡ Task C: Process partition 3
🔀 Combine: Merge results
Execution completed after 4 steps

=== Example 3: Mixed Pattern (Diamond) ===
Starting execution from: fetch
📥 Fetch: Load data
🔄 Transform A: Apply transformation A
🔄 Transform B: Apply transformation B
💾 Store: Save results
Execution completed after 4 steps
"""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def example_1_sequential():
    """Example 1: Sequential execution with >> operator."""
    print("=== Example 1: Sequential Execution ===")

    with workflow("sequential") as ctx:

        @task
        def step_1():
            print("▶️  Step 1: Fetch data")

        @task
        def step_2():
            print("▶️  Step 2: Validate data")

        @task
        def step_3():
            print("▶️  Step 3: Process data")

        # Sequential: step_1 then step_2 then step_3
        # Each step waits for the previous one to complete
        step_1 >> step_2 >> step_3

        ctx.execute("step_1")

    print()


def example_2_parallel():
    """Example 2: Parallel execution with | operator."""
    print("=== Example 2: Parallel Execution ===")

    with workflow("parallel") as ctx:

        @task
        def task_a():
            print("⚡ Task A: Process partition 1")

        @task
        def task_b():
            print("⚡ Task B: Process partition 2")

        @task
        def task_c():
            print("⚡ Task C: Process partition 3")

        @task
        def combine():
            print("🔀 Combine: Merge results")

        # Parallel: task_a, task_b, and task_c can run concurrently
        # Then combine runs after all three complete
        (task_a | task_b | task_c) >> combine

        # Note: We could start from any of the parallel tasks
        # The workflow engine will execute all of them
        ctx.execute("task_a")

    print()


def example_3_mixed():
    """Example 3: Mixed pattern - Diamond/Fan-out-Fan-in."""
    print("=== Example 3: Mixed Pattern (Diamond) ===")

    with workflow("diamond") as ctx:

        @task
        def fetch():
            """Single entry point."""
            print("📥 Fetch: Load data")

        @task
        def transform_a():
            """Parallel transformation A."""
            print("🔄 Transform A: Apply transformation A")

        @task
        def transform_b():
            """Parallel transformation B."""
            print("🔄 Transform B: Apply transformation B")

        @task
        def store():
            """Single exit point."""
            print("💾 Store: Save results")

        # Diamond pattern:
        # 1. fetch runs first
        # 2. transform_a and transform_b run in parallel (fan-out)
        # 3. store runs after both transforms complete (fan-in)
        fetch >> (transform_a | transform_b) >> store

        ctx.execute("fetch")

    print()


def main():
    """Run all operator demonstration examples."""
    example_1_sequential()
    example_2_parallel()
    example_3_mixed()
    print("All workflow operator examples completed! 🎉")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Sequential Operator (>>)**
#    Syntax: task1 >> task2
#    - task2 executes AFTER task1 completes
#    - Creates a sequential dependency chain
#    - Use for: Pipelines, ordered operations, dependent steps
#
# 2. **Parallel Operator (|)**
#    Syntax: task1 | task2
#    - task1 and task2 CAN execute concurrently
#    - No execution order guarantee between them
#    - Use for: Independent operations, partitioned data processing
#
# 3. **Combining Operators**
#    Syntax: (task1 | task2) >> task3
#    - Parentheses control precedence
#    - task3 waits for BOTH task1 AND task2
#    - Creates fan-out and fan-in patterns
#
# 4. **Common Patterns**
#
#    **Linear Pipeline**:
#    a >> b >> c >> d
#
#    **Fan-out** (one-to-many):
#    source >> (process_a | process_b | process_c)
#
#    **Fan-in** (many-to-one):
#    (input_a | input_b | input_c) >> aggregate
#
#    **Diamond** (fan-out + fan-in):
#    source >> (transform_a | transform_b) >> sink
#
#    **Multi-stage**:
#    (load_a | load_b) >> validate >> (process_a | process_b) >> store
#
# 5. **Execution Behavior**
#    - The workflow engine executes tasks in topological order
#    - Parallel tasks (|) can run simultaneously (if resources allow)
#    - Sequential tasks (>>) always maintain order
#    - Tasks start when all their dependencies complete
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Create a 3-stage pipeline with parallel tasks in each stage:
#    (load_a | load_b) >> validate >> (process_a | process_b) >> (save_db | save_file)
#
# 2. Build a map-reduce pattern:
#    source >> (map_1 | map_2 | map_3) >> reduce
#
# 3. Try a complex DAG with multiple levels:
#    a >> (b | c) >> d >> (e | f | g) >> h
#
# 4. Add timing information to see execution order:
#    import time
#    print(f"Task A at {time.time()}")
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **ETL Pipeline**:
# extract >> (validate | clean) >> transform >> (load_db | load_s3)
#
# **Data Processing**:
# fetch >> (process_region_1 | process_region_2 | process_region_3) >> aggregate
#
# **ML Pipeline**:
# load_data >> (train_model_a | train_model_b) >> ensemble >> evaluate
#
# **Web Scraping**:
# (scrape_site_1 | scrape_site_2 | scrape_site_3) >> merge >> clean >> store
#
# ============================================================================
