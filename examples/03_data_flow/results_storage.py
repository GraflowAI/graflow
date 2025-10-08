"""
Results Storage Example
========================

This example demonstrates how tasks store and retrieve execution results.
When tasks return values, Graflow automatically stores them in the channel
with the task ID as the key. Other tasks can then retrieve these results
to build data processing pipelines.

Concepts Covered:
-----------------
1. Tasks returning values (automatic result storage)
2. Retrieving task results by task ID
3. Building data pipelines with result dependencies
4. Result propagation through workflows
5. Handling missing results with defaults
6. Combining multiple task results

Expected Output:
----------------
=== Results Storage Demo ===

Starting execution from: fetch_data
📥 Fetch Data
   Fetching data from source...
   Fetched 1000 records

🔍 Validate Data
   Validating data...
   Validation result: {'valid': 950, 'invalid': 50, 'total': 1000}

🔄 Transform Data
   Retrieved fetch result: 1000 records
   Retrieved validation result: 950 valid / 50 invalid
   Transforming 950 valid records...
   Transformation complete: 950 records processed

📊 Generate Summary
   === Pipeline Summary ===
   Fetched: 1000 records
   Valid: 950 records
   Invalid: 50 records
   Processed: 950 records
   Success Rate: 95.0%

Execution completed after 4 steps

Data pipeline completed successfully! 🎉
"""

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Demonstrate task result storage and retrieval."""
    print("=== Results Storage Demo ===\n")

    with workflow("results_demo") as ctx:

        @task
        def fetch_data():
            """
            Fetch data from a source.

            When a task returns a value, Graflow automatically stores it
            in the channel using the task ID as the key.
            """
            print("📥 Fetch Data")
            print("   Fetching data from source...")

            # Simulate fetching data
            record_count = 1000
            print(f"   Fetched {record_count} records\n")

            # Return value is automatically stored in channel
            # Other tasks can retrieve it using ctx.get_result("fetch_data")
            return record_count

        @task
        def validate_data():
            """
            Validate the fetched data.

            This task also returns a value, which will be stored automatically.
            """
            print("🔍 Validate Data")
            print("   Validating data...")

            # Simulate validation
            validation_result = {
                "valid": 950,
                "invalid": 50,
                "total": 1000
            }
            print(f"   Validation result: {validation_result}\n")

            return validation_result

        @task(inject_context=True)
        def transform_data(context: TaskExecutionContext):
            """
            Transform the validated data.

            This task retrieves results from previous tasks using get_result().
            """
            print("🔄 Transform Data")

            # Retrieve result from fetch_data task
            fetch_result = context.get_result("fetch_data")
            print(f"   Retrieved fetch result: {fetch_result} records")

            # Retrieve result from validate_data task
            validation_result = context.get_result("validate_data")
            print(f"   Retrieved validation result: {validation_result['valid']} valid / {validation_result['invalid']} invalid")

            # Transform only valid records
            valid_count = validation_result["valid"]
            print(f"   Transforming {valid_count} valid records...")

            processed_count = valid_count
            print(f"   Transformation complete: {processed_count} records processed\n")

            return processed_count

        @task(inject_context=True)
        def generate_summary(context: TaskExecutionContext):
            """
            Generate a summary report using all previous results.

            This demonstrates retrieving multiple task results.
            """
            print("📊 Generate Summary")

            # Retrieve all results from previous tasks
            fetch_result = context.get_result("fetch_data")
            validation_result = context.get_result("validate_data")
            transform_result = context.get_result("transform_data")

            # Calculate success rate
            success_rate = (validation_result["valid"] / validation_result["total"]) * 100

            # Generate summary
            print("   === Pipeline Summary ===")
            print(f"   Fetched: {fetch_result} records")
            print(f"   Valid: {validation_result['valid']} records")
            print(f"   Invalid: {validation_result['invalid']} records")
            print(f"   Processed: {transform_result} records")
            print(f"   Success Rate: {success_rate}%\n")

            return {
                "fetched": fetch_result,
                "valid": validation_result["valid"],
                "processed": transform_result,
                "success_rate": success_rate
            }

        # Define workflow: fetch -> validate -> transform -> summary
        # Note: fetch and validate run sequentially, then transform uses both results
        fetch_data >> validate_data >> transform_data >> generate_summary

        # Execute
        ctx.execute("fetch_data")

    print("Data pipeline completed successfully! 🎉")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Automatic Result Storage**
#    @task
#    def my_task():
#        return result  # Automatically stored with key = task ID
#
#    - Tasks that return values have results automatically stored
#    - Storage happens after successful task execution
#    - Result is stored in channel with task ID as key
#
# 2. **Retrieving Results**
#    result = context.get_result("task_id")
#
#    - Use get_result() to retrieve any task's result
#    - Task must have completed before you retrieve its result
#    - Workflow execution order ensures dependencies are met
#
# 3. **Default Values**
#    result = context.get_result("task_id", default=None)
#
#    - Provide default if result might not exist
#    - Useful for optional tasks or error handling
#    - Prevents KeyError exceptions
#
# 4. **Result Types**
#    - Can return any Python object
#    - Common types: dict, list, int, str, custom objects
#    - Results persist for the entire workflow execution
#
# 5. **Result Dependencies**
#    task1 >> task2  # task2 can access task1's result
#
#    - Use >> to ensure execution order
#    - Dependent task only runs after dependency completes
#    - Results are available when dependent task runs
#
# 6. **Multiple Results**
#    result1 = ctx.get_result("task1")
#    result2 = ctx.get_result("task2")
#    combined = process(result1, result2)
#
#    - Tasks can retrieve results from multiple previous tasks
#    - Useful for aggregation and summary tasks
#    - Build complex data pipelines
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Create a fan-in pattern (multiple tasks → one aggregator):
#    @task
#    def source_a():
#        return [1, 2, 3]
#
#    @task
#    def source_b():
#        return [4, 5, 6]
#
#    @task(inject_context=True)
#    def combine(ctx: TaskExecutionContext):
#        a = ctx.get_result("source_a")
#        b = ctx.get_result("source_b")
#        return a + b
#
#    (source_a | source_b) >> combine
#
# 2. Handle missing results:
#    result = ctx.get_result("optional_task", default={"empty": True})
#    if result.get("empty"):
#        print("Optional task did not run")
#
# 3. Chain transformations:
#    raw = ctx.get_result("fetch")
#    cleaned = ctx.get_result("clean")
#    transformed = ctx.get_result("transform")
#    # Each task builds on the previous
#
# 4. Return complex objects:
#    return {
#        "data": [...],
#        "metadata": {"count": 100, "source": "db"},
#        "status": "success"
#    }
#
# 5. Calculate derived metrics:
#    @task(inject_context=True)
#    def calculate_metrics(ctx: TaskExecutionContext):
#        counts = [ctx.get_result(f"task_{i}") for i in range(5)]
#        return {"total": sum(counts), "average": sum(counts) / len(counts)}
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **ETL Pipeline**:
# extract() returns raw data
# transform() gets extract result, returns cleaned data
# load() gets transform result, loads to destination
#
# **Data Validation Pipeline**:
# fetch() returns dataset
# validate() gets dataset, returns validation report
# filter() gets validation report, returns clean data
# store() gets clean data, persists it
#
# **Multi-Source Aggregation**:
# fetch_db() returns DB data
# fetch_api() returns API data
# fetch_file() returns file data
# merge() gets all three, returns combined dataset
#
# **Report Generation**:
# run_analysis() returns metrics
# generate_chart() gets metrics, returns chart
# create_report() gets metrics and chart, returns PDF
#
# **Machine Learning Pipeline**:
# load_data() returns dataset
# preprocess() gets dataset, returns features
# train() gets features, returns model
# evaluate() gets model, returns metrics
#
# ============================================================================
# Pattern: Result vs Channel
# ============================================================================
#
# **Use get_result() when**:
# ✅ Direct task-to-task data flow
# ✅ Task return values are the primary output
# ✅ Linear or tree-like dependencies
# ✅ Each task processes previous task's output
#
# **Use channel.set/get when**:
# ✅ Shared state across multiple tasks
# ✅ Configuration or context that many tasks need
# ✅ Accumulating values (counters, logs)
# ✅ Broadcasting data to multiple consumers
#
# **Combine both**:
# @task(inject_context=True)
# def my_task(ctx: TaskExecutionContext):
#     # Get input from previous task
#     input_data = ctx.get_result("previous_task")
#
#     # Get shared config from channel
#     config = ctx.get_channel().get("config")
#
#     # Process with both
#     result = process(input_data, config)
#
#     # Return result (auto-stored)
#     return result
#
# ============================================================================
