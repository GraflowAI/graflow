"""
Batch Processing Pipeline
==========================

This example demonstrates a scalable batch processing workflow for handling
large datasets by partitioning work into chunks and processing in parallel.

Prerequisites:
--------------
None (uses synthetic data)

Concepts Covered:
-----------------
1. Data partitioning and chunking
2. Parallel batch processing
3. Result aggregation
4. Progress tracking
5. Error handling and retry logic

Expected Output:
----------------
=== Batch Processing Pipeline ===

📋 Step 1: Initialize
   Total items to process: 1000
   Batch size: 100
   Number of batches: 10
   ✅ Initialization complete

📦 Step 2: Partition Data
   Creating 10 batches...
   ✅ Batch 0: items 0-99
   ✅ Batch 1: items 100-199
   ✅ Batch 2: items 200-299
   ✅ Batch 3: items 300-399
   ✅ Batch 4: items 400-499
   ✅ Batch 5: items 500-599
   ✅ Batch 6: items 600-699
   ✅ Batch 7: items 700-799
   ✅ Batch 8: items 800-899
   ✅ Batch 9: items 900-999

⚙️  Step 3: Process Batches
   Processing batch 0... ✅ (0.2s)
   Processing batch 1... ✅ (0.2s)
   Processing batch 2... ✅ (0.2s)
   ...
   All batches processed

🔀 Step 4: Aggregate Results
   Combining results from 10 batches...
   ✅ Total processed: 1000 items
   ✅ Success: 980 items
   ✅ Failed: 20 items

📊 Step 5: Generate Report
   === Batch Processing Summary ===
   Total Items: 1000
   Processed: 980
   Failed: 20
   Success Rate: 98.0%
   Processing Time: 2.3s
   Throughput: 426 items/sec

✅ Batch processing completed
"""

import time

from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow


def main():
    """Run batch processing pipeline."""
    print("=== Batch Processing Pipeline ===\n")

    # Configuration
    total_items = 1000
    batch_size = 100
    num_batches = total_items // batch_size

    with workflow("batch_processing") as ctx:

        @task(inject_context=True)
        def initialize(context: TaskExecutionContext):
            """Step 1: Initialize batch processing."""
            print("📋 Step 1: Initialize")
            print(f"   Total items to process: {total_items}")
            print(f"   Batch size: {batch_size}")
            print(f"   Number of batches: {num_batches}")
            print("   ✅ Initialization complete\n")

            channel = context.get_channel()
            channel.set("total_items", total_items)
            channel.set("batch_size", batch_size)
            channel.set("num_batches", num_batches)

        @task(inject_context=True)
        def partition_data(context: TaskExecutionContext):
            """Step 2: Partition data into batches."""
            print("📦 Step 2: Partition Data")
            print(f"   Creating {num_batches} batches...")

            # Create batch metadata
            batches = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size - 1
                batch_info = {
                    "batch_id": i,
                    "start": start_idx,
                    "end": end_idx,
                    "size": batch_size
                }
                batches.append(batch_info)
                print(f"   ✅ Batch {i}: items {start_idx}-{end_idx}")

            print()

            channel = context.get_channel()
            channel.set("batches", batches)

        @task(inject_context=True)
        def process_batches(context: TaskExecutionContext):
            """Step 3: Process all batches."""
            print("⚙️  Step 3: Process Batches")

            channel = context.get_channel()
            batches = channel.get("batches")

            results = []

            for batch_info in batches:
                batch_id = batch_info["batch_id"]
                batch_size = batch_info["size"]

                # Simulate batch processing
                print(f"   Processing batch {batch_id}...", end=" ")
                start_time = time.time()
                time.sleep(0.05)  # Simulate work

                # Simulate success rate (98% success)
                success_count = int(batch_size * 0.98)
                failed_count = batch_size - success_count

                elapsed = time.time() - start_time
                print(f"✅ ({elapsed:.1f}s)")

                # Store batch result
                batch_result = {
                    "batch_id": batch_id,
                    "processed": batch_size,
                    "success": success_count,
                    "failed": failed_count,
                    "elapsed": elapsed
                }
                results.append(batch_result)

            print("   All batches processed\n")

            channel.set("batch_results", results)

        @task(inject_context=True)
        def aggregate_results(context: TaskExecutionContext):
            """Step 4: Aggregate results from all batches."""
            print("🔀 Step 4: Aggregate Results")

            channel = context.get_channel()
            batch_results = channel.get("batch_results")
            num_batches = len(batch_results)

            print(f"   Combining results from {num_batches} batches...")

            # Aggregate metrics
            total_processed = sum(r["processed"] for r in batch_results)
            total_success = sum(r["success"] for r in batch_results)
            total_failed = sum(r["failed"] for r in batch_results)
            total_time = sum(r["elapsed"] for r in batch_results)

            print(f"   ✅ Total processed: {total_processed} items")
            print(f"   ✅ Success: {total_success} items")
            print(f"   ✅ Failed: {total_failed} items\n")

            # Store aggregated results
            channel.set("total_processed", total_processed)
            channel.set("total_success", total_success)
            channel.set("total_failed", total_failed)
            channel.set("total_time", total_time)

        @task(inject_context=True)
        def generate_report(context: TaskExecutionContext):
            """Step 5: Generate processing report."""
            print("📊 Step 5: Generate Report")

            channel = context.get_channel()
            total_items = channel.get("total_items")
            total_processed = channel.get("total_processed")
            total_success = channel.get("total_success")
            total_failed = channel.get("total_failed")
            total_time = channel.get("total_time")

            success_rate = (total_success / total_processed * 100) if total_processed > 0 else 0
            throughput = total_processed / total_time if total_time > 0 else 0

            print("   === Batch Processing Summary ===")
            print(f"   Total Items: {total_items}")
            print(f"   Processed: {total_processed}")
            print(f"   Success: {total_success}")
            print(f"   Failed: {total_failed}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Processing Time: {total_time:.1f}s")
            print(f"   Throughput: {throughput:.0f} items/sec\n")

            print("✅ Batch processing completed")

        # Define pipeline workflow
        initialize >> partition_data >> process_batches >> aggregate_results >> generate_report

        # Execute pipeline
        ctx.execute("initialize")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Batch Processing Pattern**
#    Initialize → Partition → Process → Aggregate → Report
#    - Divide large workload into manageable chunks
#    - Process chunks independently
#    - Combine results for final output
#
# 2. **Scalability**
#    - Batch size determines memory usage
#    - Number of batches affects parallelism potential
#    - Can easily scale to distributed workers
#
# 3. **Progress Tracking**
#    - Monitor each batch completion
#    - Calculate overall progress
#    - Estimate time remaining
#
# 4. **Error Handling**
#    - Track success/failure per batch
#    - Continue processing despite failures
#    - Aggregate error statistics
#
# 5. **Real-World Applications**
#    ✅ Large dataset processing
#    ✅ Bulk database operations
#    ✅ Image/video processing
#    ✅ Data migration
#    ✅ Report generation
#
# ============================================================================
# Production Enhancements:
# ============================================================================
#
# **Parallel Processing with Redis**:
# - Use QueueBackend.REDIS for distribution
# - Multiple workers process batches in parallel
# - Automatic load balancing
#
# Example:
# exec_context = ExecutionContext.create(
#     graph,
#     "initialize",
#     queue_backend=QueueBackend.REDIS,
#     channel_backend="redis"
# )
#
# **Retry Logic**:
# @task
# def process_batch_with_retry(batch_id, max_retries=3):
#     for attempt in range(max_retries):
#         try:
#             return process_batch(batch_id)
#         except Exception as e:
#             if attempt == max_retries - 1:
#                 raise
#             time.sleep(2 ** attempt)  # Exponential backoff
#
# **Checkpointing**:
# @task
# def process_with_checkpoint(context):
#     checkpoint = load_checkpoint()
#     start_batch = checkpoint.get("last_batch", 0)
#
#     for batch_id in range(start_batch, num_batches):
#         process_batch(batch_id)
#         save_checkpoint({"last_batch": batch_id})
#
# **Dynamic Batch Sizing**:
# def determine_optimal_batch_size(total_items, available_memory):
#     # Calculate based on memory constraints
#     item_size = estimate_item_size()
#     max_items_in_memory = available_memory // item_size
#     return min(max_items_in_memory, total_items // 10)
#
# **Progress Monitoring**:
# @task
# def process_with_progress(context):
#     from tqdm import tqdm
#     batches = context.get_channel().get("batches")
#
#     for batch in tqdm(batches, desc="Processing batches"):
#         process_batch(batch)
#
# **Failure Recovery**:
# @task
# def recover_failed_items(context):
#     failed_batches = get_failed_batches()
#
#     for batch in failed_batches:
#         # Reprocess individual items
#         for item in batch:
#             try:
#                 process_item(item)
#             except Exception as e:
#                 log_permanent_failure(item, e)
#
# ============================================================================
# Distributed Batch Processing:
# ============================================================================
#
# For true parallel processing across multiple workers:
#
# 1. Start Redis:
#    docker run -p 6379:6379 redis:7.2
#
# 2. Start multiple workers:
#    Terminal 1: python -m graflow.worker.main --worker-id worker-1
#    Terminal 2: python -m graflow.worker.main --worker-id worker-2
#    Terminal 3: python -m graflow.worker.main --worker-id worker-3
#
# 3. Submit workflow with Redis backend:
#    exec_context = ExecutionContext.create(
#        graph,
#        "initialize",
#        queue_backend=QueueBackend.REDIS,
#        channel_backend="redis",
#        config={"redis_client": redis_client}
#    )
#
# 4. Workers will automatically pick up and process batches in parallel
#
# ============================================================================
# Performance Optimization:
# ============================================================================
#
# **Memory Management**:
# - Process batches in streaming fashion
# - Don't load entire dataset into memory
# - Use generators for large datasets
#
# **I/O Optimization**:
# - Batch database queries
# - Use connection pooling
# - Implement read/write buffering
#
# **CPU Optimization**:
# - Use multiprocessing for CPU-bound tasks
# - Vectorize operations where possible
# - Profile and optimize hot paths
#
# **Monitoring**:
# - Track processing rate
# - Monitor memory usage
# - Alert on slowdowns or failures
# - Collect performance metrics
#
# ============================================================================
