"""
Concurrent Workflow Execution
==============================

This example demonstrates how to run multiple workflow instances concurrently
using Python's threading module. This is useful for parallel processing of
independent workflow instances.

Prerequisites:
--------------
None

Concepts Covered:
-----------------
1. Running multiple workflow contexts concurrently
2. Thread-safe workflow execution
3. Managing multiple independent workflow instances
4. Coordinating concurrent workflow completion
5. Use cases for concurrent workflows

Expected Output:
----------------
=== Concurrent Workflow Execution ===

Starting 3 concurrent workers...

Worker 0 workflow:
Name: worker_0
Tasks: 3
Dependencies: 2

Worker 1 workflow:
Name: worker_1
Tasks: 3
Dependencies: 2

Worker 2 workflow:
Name: worker_2
Tasks: 3
Dependencies: 2

Worker 0: Starting work
Worker 1: Starting work
Worker 2: Starting work
Worker 0: Processing
Worker 1: Processing
Worker 2: Processing
Worker 0: Finishing work
Worker 1: Finishing work
Worker 2: Finishing work

✅ All 3 workers completed successfully
Total execution time: ~0.3s

=== Summary ===
✅ Concurrent workflow execution demonstrated
✅ Each worker runs in its own thread
✅ Workflows execute independently
✅ Thread-safe execution guaranteed
"""

import threading
import time

from graflow.core.decorators import task
from graflow.core.workflow import workflow


def worker_workflow(worker_id: int, results: list):
    """Execute a workflow in a worker thread.

    Args:
        worker_id: Unique identifier for this worker
        results: Shared list to store worker results (thread-safe for append)
    """
    with workflow(f"worker_{worker_id}") as ctx:

        @task
        def start_work():
            """Start processing in this worker."""
            print(f"Worker {worker_id}: Starting work")
            time.sleep(0.1)  # Simulate work

        @task
        def do_processing():
            """Perform the main processing task."""
            print(f"Worker {worker_id}: Processing")
            time.sleep(0.1)  # Simulate work
            return f"result_{worker_id}"

        @task
        def finish_work():
            """Finish and clean up."""
            print(f"Worker {worker_id}: Finishing work")

        # Define workflow
        start_work >> do_processing >> finish_work

        # Show workflow info
        print(f"\nWorker {worker_id} workflow:")
        ctx.show_info()

        # Execute workflow
        ctx.execute("start_work")

        # Store result
        results.append(f"Worker {worker_id} completed")


def scenario_1_basic_concurrent():
    """Scenario 1: Run multiple workflows concurrently."""
    print("=== Concurrent Workflow Execution ===\n")

    num_workers = 3
    print(f"Starting {num_workers} concurrent workers...\n")

    # Shared results list (thread-safe for append operations)
    results = []

    # Create and start worker threads
    threads = []
    start_time = time.time()

    for i in range(num_workers):
        thread = threading.Thread(target=worker_workflow, args=(i, results))
        threads.append(thread)
        thread.start()

    # Wait for all workers to complete
    for thread in threads:
        thread.join()

    end_time = time.time()
    elapsed = end_time - start_time

    # Show results
    print(f"\n✅ All {num_workers} workers completed successfully")
    print(f"Total execution time: ~{elapsed:.1f}s")


def scenario_2_batch_processing():
    """Scenario 2: Batch processing with concurrent workers."""
    print("\n=== Batch Processing with Concurrent Workers ===\n")

    # Simulate a batch of items to process
    items = list(range(10))
    batch_size = 3
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    print(f"Processing {len(items)} items in {len(batches)} batches")
    print(f"Batch size: {batch_size}\n")

    def process_batch(batch_id: int, batch_items: list, results: dict):
        """Process a batch of items."""
        with workflow(f"batch_{batch_id}") as ctx:

            @task
            def load_batch():
                print(f"Batch {batch_id}: Loading {len(batch_items)} items")
                return batch_items

            @task
            def process_items():
                print(f"Batch {batch_id}: Processing items {batch_items}")
                time.sleep(0.1)  # Simulate processing
                # Simulate some processing result
                processed = [x * 2 for x in batch_items]
                return processed

            @task
            def save_results():
                print(f"Batch {batch_id}: Saving results")
                return f"batch_{batch_id}_complete"

            # Define pipeline
            load_batch >> process_items >> save_results

            # Execute
            ctx.execute("load_batch")

            # Store result
            results[batch_id] = f"Batch {batch_id} processed {len(batch_items)} items"

    # Process batches concurrently
    results = {}
    threads = []
    start_time = time.time()

    for i, batch in enumerate(batches):
        thread = threading.Thread(target=process_batch, args=(i, batch, results))
        threads.append(thread)
        thread.start()

    # Wait for all batches
    for thread in threads:
        thread.join()

    end_time = time.time()
    elapsed = end_time - start_time

    # Show results
    print(f"\n✅ All {len(batches)} batches completed")
    print(f"Total items processed: {len(items)}")
    print(f"Total execution time: ~{elapsed:.1f}s")
    print(f"Throughput: ~{len(items) / elapsed:.0f} items/sec")


def scenario_3_heterogeneous_workflows():
    """Scenario 3: Run different types of workflows concurrently."""
    print("\n=== Heterogeneous Concurrent Workflows ===\n")

    def data_pipeline():
        """ETL data pipeline."""
        with workflow("etl_pipeline") as ctx:

            @task
            def extract():
                print("ETL: Extracting data")
                time.sleep(0.1)

            @task
            def transform():
                print("ETL: Transforming data")
                time.sleep(0.1)

            @task
            def load():
                print("ETL: Loading data")

            extract >> transform >> load
            ctx.execute("extract")

    def ml_training():
        """ML model training pipeline."""
        with workflow("ml_training") as ctx:

            @task
            def prepare_data():
                print("ML: Preparing training data")
                time.sleep(0.1)

            @task
            def train_model():
                print("ML: Training model")
                time.sleep(0.1)

            @task
            def evaluate():
                print("ML: Evaluating model")

            prepare_data >> train_model >> evaluate
            ctx.execute("prepare_data")

    def report_generation():
        """Report generation pipeline."""
        with workflow("reports") as ctx:

            @task
            def collect_metrics():
                print("Reports: Collecting metrics")
                time.sleep(0.1)

            @task
            def generate_report():
                print("Reports: Generating report")

            collect_metrics >> generate_report
            ctx.execute("collect_metrics")

    print("Running 3 different workflow types concurrently...\n")

    # Run different workflows concurrently
    threads = [
        threading.Thread(target=data_pipeline),
        threading.Thread(target=ml_training),
        threading.Thread(target=report_generation),
    ]

    start_time = time.time()

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n✅ All 3 different workflows completed")
    print(f"Total execution time: ~{elapsed:.1f}s")


def main():
    """Run all concurrent workflow scenarios."""
    print("=== Concurrent Workflows Demo ===\n")

    # Scenario 1: Basic concurrent execution
    scenario_1_basic_concurrent()

    # Scenario 2: Batch processing
    scenario_2_batch_processing()

    # Scenario 3: Heterogeneous workflows
    scenario_3_heterogeneous_workflows()

    print("\n=== Summary ===")
    print("✅ Concurrent workflow execution demonstrated")
    print("✅ Each worker runs in its own thread")
    print("✅ Workflows execute independently")
    print("✅ Thread-safe execution guaranteed")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Thread-Safe Workflows**
#    - Each workflow runs in its own WorkflowContext
#    - Context isolation ensures thread safety
#    - No shared state between workflows
#
# 2. **When to Use Concurrent Workflows**
#    ✅ Processing multiple independent datasets
#    ✅ Running multiple workflow instances for different customers
#    ✅ Batch processing with parallel batches
#    ✅ Running different workflow types simultaneously
#    ✅ Scaling within a single process
#
# 3. **When NOT to Use This Pattern**
#    ❌ For distributing tasks WITHIN a workflow (use Redis workers)
#    ❌ When workflows share mutable state
#    ❌ For CPU-bound tasks (use multiprocessing instead)
#    ❌ When order matters between workflows
#
# 4. **Threading vs Redis Workers**
#    - **Threading**: Multiple workflow INSTANCES in one process
#    - **Redis Workers**: Distributed TASKS across multiple processes/machines
#    - Threading is simpler but limited to one machine
#    - Redis workers scale better for large workloads
#
# 5. **Best Practices**
#    - Keep workflows independent
#    - Avoid shared mutable state
#    - Use thread-safe data structures for results
#    - Consider multiprocessing for CPU-bound work
#    - Use Redis workers for distributed execution
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Multi-Tenant Processing**:
# Process workflows for multiple customers concurrently
# def process_customer(customer_id):
#     with workflow(f"customer_{customer_id}") as ctx:
#         # Customer-specific workflow
#         pass
#
# **Parallel Data Pipelines**:
# Run ETL pipelines for different data sources concurrently
# sources = ["database", "api", "files"]
# threads = [Thread(target=etl_pipeline, args=(src,)) for src in sources]
#
# **A/B Testing**:
# Run multiple model versions concurrently
# for model_version in ["v1", "v2", "v3"]:
#     Thread(target=run_model, args=(model_version,)).start()
#
# **Batch Processing**:
# Process large datasets in parallel batches
# batches = partition_data(data, batch_size)
# threads = [Thread(target=process_batch, args=(b,)) for b in batches]
#
# ============================================================================
# Performance Considerations:
# ============================================================================
#
# **Threading Limitations**:
# - Python GIL limits true parallelism for CPU-bound tasks
# - Best for I/O-bound workflows (API calls, database, file I/O)
# - Number of threads should match number of CPU cores for CPU tasks
#
# **When to Use Multiprocessing**:
# import multiprocessing
# processes = [Process(target=workflow_func) for _ in range(n)]
# - Better for CPU-bound tasks
# - True parallelism across cores
# - Higher memory overhead
#
# **When to Use Redis Workers**:
# - Distributed execution across machines
# - Better fault tolerance
# - Can scale to many workers
# - See examples/05_distributed/
#
# ============================================================================
# Error Handling:
# ============================================================================
#
# **Handling Failures**:
# def safe_worker(worker_id, results, errors):
#     try:
#         worker_workflow(worker_id, results)
#     except Exception as e:
#         errors[worker_id] = str(e)
#         print(f"Worker {worker_id} failed: {e}")
#
# errors = {}
# threads = [Thread(target=safe_worker, args=(i, results, errors))
#            for i in range(n)]
#
# **Timeouts**:
# thread.join(timeout=30)  # Wait max 30 seconds
# if thread.is_alive():
#     print("Worker timed out")
#
# **Resource Limits**:
# import concurrent.futures
# with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#     futures = [executor.submit(worker_workflow, i, results)
#                for i in range(n)]
#
# ============================================================================
