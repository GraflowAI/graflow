"""
Dynamic Task Generation
=======================

This example demonstrates how to dynamically generate and add tasks to a workflow
based on runtime conditions or configuration.

Prerequisites:
--------------
None

Concepts Covered:
-----------------
1. Dynamic task creation
2. Runtime workflow modification
3. Conditional task execution
4. Task factory patterns
5. Graph manipulation

Expected Output:
----------------
=== Dynamic Task Generation ===

Scenario 1: Batch Processing
Creating 5 processing tasks dynamically...
✅ Task process_item_0 created
✅ Task process_item_1 created
✅ Task process_item_2 created
✅ Task process_item_3 created
✅ Task process_item_4 created

Executing batch workflow:
Processing item 0
Processing item 1
Processing item 2
Processing item 3
Processing item 4
Aggregating 5 results

Scenario 2: Conditional Pipeline
Configuration: enable_validation=True, enable_transform=True
Building conditional pipeline...
✅ Extract task added
✅ Validation task added (conditional)
✅ Transform task added (conditional)
✅ Load task added

Executing conditional pipeline:
Extracting data...
Validating data...
Transforming data...
Loading data...

=== Summary ===
✅ Dynamic tasks created based on runtime config
✅ Conditional workflow branching implemented
✅ Task factories demonstrated
"""

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def scenario_1_batch_processing():
    """Scenario 1: Generate tasks for batch processing."""
    print("Scenario 1: Batch Processing")
    print("Creating 5 processing tasks dynamically...")

    with workflow("batch_processing") as ctx:
        # Create tasks dynamically based on number of items
        num_items = 5
        process_tasks = []

        for i in range(num_items):
            # Create a closure that captures the item index
            def make_processor(item_id):
                # Use unique task ID for each item
                @task(id=f"process_item_{item_id}")
                def process_item():
                    print(f"Processing item {item_id}")
                    return f"result_{item_id}"

                return process_item

            # Create and register the task
            process_task = make_processor(i)
            process_tasks.append(process_task)
            print(f"✅ Task process_item_{i} created")

        @task
        def aggregate_results():
            print(f"Aggregating {num_items} results")
            return f"aggregated_{num_items}_results"

        # Connect all processing tasks to aggregation
        # All process tasks run in parallel, then aggregate
        for process_task in process_tasks:
            process_task >> aggregate_results

        print("\nExecuting batch workflow:")
        ctx.execute("process_item_0")

    print()


def scenario_2_conditional_pipeline():
    """Scenario 2: Build pipeline based on configuration."""
    print("Scenario 2: Conditional Pipeline")

    # Runtime configuration
    config = {
        "enable_validation": True,
        "enable_transform": True,
        "enable_caching": False,
    }

    print(
        f"Configuration: enable_validation={config['enable_validation']}, enable_transform={config['enable_transform']}"
    )
    print("Building conditional pipeline...")

    with workflow("conditional_pipeline") as ctx:

        @task
        def extract_data():
            print("Extracting data...")
            return {"records": 100}

        print("✅ Extract task added")

        # Conditionally add validation
        if config["enable_validation"]:

            @task
            def validate_data():
                print("Validating data...")
                return {"valid": True}

            extract_data >> validate_data
            last_task = validate_data
            print("✅ Validation task added (conditional)")
        else:
            last_task = extract_data

        # Conditionally add transformation
        if config["enable_transform"]:

            @task
            def transform_data():
                print("Transforming data...")
                return {"transformed": True}

            last_task >> transform_data
            last_task = transform_data
            print("✅ Transform task added (conditional)")

        # Always add load
        @task
        def load_data():
            print("Loading data...")
            return {"loaded": True}

        last_task >> load_data
        print("✅ Load task added")

        print("\nExecuting conditional pipeline:")
        ctx.execute("extract_data")

    print()


def scenario_3_task_factory():
    """Scenario 3: Use factory pattern for task creation."""
    print("Scenario 3: Task Factory Pattern")

    def create_calculator_task(operation: str, value: int):
        """Factory function that creates calculator tasks."""

        # Use unique task ID based on operation and value
        task_id = f"calc_{operation}_{value}"

        if operation == "add":

            @task(id=task_id)
            def calculator():
                result = 10 + value
                print(f"Add: 10 + {value} = {result}")
                return result
        elif operation == "multiply":

            @task(id=task_id)
            def calculator():
                result = 10 * value
                print(f"Multiply: 10 * {value} = {result}")
                return result
        elif operation == "power":

            @task(id=task_id)
            def calculator():
                result = 10**value
                print(f"Power: 10 ** {value} = {result}")
                return result
        else:

            @task(id=task_id)
            def calculator():
                print(f"Unknown operation: {operation}")

        return calculator

    with workflow("task_factory") as ctx:
        # Create tasks from factory
        operations = [
            ("add", 5),
            ("multiply", 3),
            ("power", 2),
        ]

        print(f"Creating {len(operations)} calculator tasks...")

        tasks = []
        for op, val in operations:
            calc_task = create_calculator_task(op, val)
            tasks.append(calc_task)
            print(f"✅ Created {op} task with value {val}")

        @task
        def summarize():
            print("All calculations complete")
            return "done"

        # All calculations run in parallel, then summarize
        for calc_task in tasks:
            calc_task >> summarize

        print("\nExecuting calculations:")
        ctx.execute("calc_add_5")

    print()


def main():
    """Run all dynamic task scenarios."""
    print("=== Dynamic Task Generation ===\n")

    # Scenario 1: Batch processing with dynamic task count
    scenario_1_batch_processing()

    # Scenario 2: Conditional pipeline based on config
    scenario_2_conditional_pipeline()

    # Scenario 3: Task factory pattern
    scenario_3_task_factory()

    print("=== Summary ===")
    print("✅ Dynamic tasks created based on runtime config")
    print("✅ Conditional workflow branching implemented")
    print("✅ Task factories demonstrated")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Dynamic Task Creation**
#    - Tasks can be created at runtime
#    - Use loops to generate multiple similar tasks
#    - Closures capture runtime values
#
# 2. **Factory Pattern**
#    - Create parameterized task generators
#    - Encapsulate task creation logic
#    - Reusable task templates
#
# 3. **Conditional Workflows**
#    - Build workflow based on configuration
#    - Add/skip tasks based on conditions
#    - Dynamic pipeline composition
#
# 4. **Use Cases**
#    ✅ Batch processing variable number of items
#    ✅ Feature flags and A/B testing
#    ✅ Environment-specific pipelines
#    ✅ Data-driven workflow generation
#
# 5. **Best Practices**
#    - Use factory functions for reusable patterns
#    - Keep closures simple
#    - Document dynamic behavior
#    - Test with different configurations
#
# ============================================================================
# Advanced Patterns:
# ============================================================================
#
# **Data-Driven Task Generation**:
# config_list = load_from_file("tasks.yaml")
# for task_config in config_list:
#     create_task_from_config(task_config)
#
# **Dynamic Parallelism**:
# chunk_size = determine_optimal_chunk_size(data)
# chunks = partition_data(data, chunk_size)
# tasks = [create_processor(chunk) for chunk in chunks]
#
# **Recursive Task Generation**:
# def create_recursive_tasks(depth):
#     if depth == 0:
#         return base_task()
#     return create_recursive_tasks(depth - 1) >> process_task()
#
# ============================================================================
