"""
Distributed Workflow Example
==============================

This example demonstrates a complete distributed ETL workflow executed across
multiple workers using Redis as the coordination backend.

Prerequisites:
--------------
1. Redis running: docker run -p 6379:6379 redis:7.2
2. pip install redis
3. Start 2-3 workers in separate terminals:
   python -m graflow.worker.main --worker-id worker-1
   python -m graflow.worker.main --worker-id worker-2

Concepts Covered:
-----------------
1. Complete distributed workflow
2. Task distribution across workers
3. Result sharing via Redis
4. Parallel processing with multiple workers
5. Workflow completion monitoring

Expected Output:
----------------
=== Distributed ETL Workflow ===

Step 1: Setup
✅ Redis connected
✅ 3 extraction tasks created
✅ 1 aggregation task created

Step 2: Submitting workflow to Redis
✅ Workflow submitted
✅ Workers will process tasks...

Step 3: Monitoring execution (this may take 10-15 seconds)
⏳ Waiting for workers to process tasks...
✅ extract_source_1 completed
✅ extract_source_2 completed
✅ extract_source_3 completed
✅ aggregate_results completed

Step 4: Results
Source 1: 1000 records
Source 2: 1500 records
Source 3: 800 records
Total aggregated: 3300 records

=== Summary ===
✅ Distributed workflow completed
✅ 3 extraction tasks processed in parallel
✅ 1 aggregation task processed results
✅ All tasks distributed across workers
"""

import random
import sys
import time


def check_redis():
    """Check Redis availability."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        print(f"❌ Redis not available: {e}")
        return None


def main():
    """Run distributed workflow demonstration."""
    print("=== Distributed ETL Workflow ===\n")

    # Step 1: Setup
    print("Step 1: Setup")
    redis_client = check_redis()
    if not redis_client:
        print("\n⚠️  Start Redis: docker run -p 6379:6379 redis:7.2")
        sys.exit(1)

    print("✅ Redis connected")

    from graflow.core.context import ExecutionContext
    from graflow.core.decorators import task
    from graflow.core.workflow import workflow
    from graflow.queue.distributed import DistributedTaskQueue

    # Create workflow
    with workflow("distributed_etl") as ctx:

        @task
        def extract_source_1():
            """Extract data from source 1."""
            time.sleep(random.uniform(1, 2))  # Simulate work
            return {"source": "source_1", "records": 1000}

        @task
        def extract_source_2():
            """Extract data from source 2."""
            time.sleep(random.uniform(1, 2))
            return {"source": "source_2", "records": 1500}

        @task
        def extract_source_3():
            """Extract data from source 3."""
            time.sleep(random.uniform(1, 2))
            return {"source": "source_3", "records": 800}

        @task(inject_context=True)
        def aggregate_results(context):
            """Aggregate results from all sources."""

            # Get results from all extraction tasks
            result1 = context.get_result("extract_source_1")
            result2 = context.get_result("extract_source_2")
            result3 = context.get_result("extract_source_3")

            # Aggregate
            total = result1["records"] + result2["records"] + result3["records"]

            return {
                "sources": [result1, result2, result3],
                "total_records": total
            }

        # Define parallel extraction followed by aggregation
        (extract_source_1 | extract_source_2 | extract_source_3) >> aggregate_results

        print("✅ 3 extraction tasks created")
        print("✅ 1 aggregation task created")

        # Step 2: Submit workflow
        print("\nStep 2: Submitting workflow to Redis")

        # Create execution context with Redis
        exec_context = ExecutionContext.create(
            ctx.graph,
            "extract_source_1",
            channel_backend="redis",
            max_steps=20,
            config={"redis_client": redis_client}
        )

        redis_queue = DistributedTaskQueue(
            redis_client=redis_client,
            key_prefix="graflow:distributed_demo"
        )

        redis_queue.cleanup()
        exec_context.channel.clear()

        # Submit tasks to Redis queue
        # Note: In production, WorkflowEngine.execute() would manage scheduling.
        # Here we demonstrate manual enqueueing for clarity.
        from graflow.queue.base import TaskSpec

        # Tasks are stored in Graph, no need to register separately
        # Workers will retrieve tasks from Graph via GraphStore

        # Enqueue all extraction tasks (they can run in parallel)
        # Tasks are retrieved from Graph (via GraphStore), no serialization strategy needed
        for task_id in ["extract_source_1", "extract_source_2", "extract_source_3"]:
            task_node = ctx.graph.get_node(task_id)
            task_spec = TaskSpec(
                executable=task_node,
                execution_context=exec_context
            )
            redis_queue.enqueue(task_spec)

        print("✅ Workflow submitted")
        print("✅ Workers will process tasks...")
        print("\n⚠️  Make sure workers are running with matching prefix:")
        print("   python -m graflow.worker.main --worker-id worker-1 --redis-key-prefix graflow:distributed_demo")
        print("   python -m graflow.worker.main --worker-id worker-2 --redis-key-prefix graflow:distributed_demo")

        # Step 3: Monitor execution
        print("\nStep 3: Monitoring execution (this may take 10-15 seconds)")
        print("⏳ Waiting for workers to process tasks...")

        # Wait for extraction tasks to complete
        completed = set()
        for _ in range(30):  # Wait up to 30 seconds
            time.sleep(1)

            for task_id in ["extract_source_1", "extract_source_2", "extract_source_3"]:
                if task_id not in completed:
                    result = exec_context.get_result(task_id)
                    if result is not None:
                        print(f"✅ {task_id} completed")
                        completed.add(task_id)

            if len(completed) == 3:
                break

        if len(completed) < 3:
            print(f"\n⚠️  Not all extraction tasks completed. Finished tasks: {completed}")
            print("   Make sure workers are running!")
            sys.exit(1)

        # Enqueue aggregation task
        agg_task = ctx.graph.get_node("aggregate_results")
        agg_spec = TaskSpec(
            executable=agg_task,
            execution_context=exec_context
        )
        print("✅ Extraction phase complete, enqueuing aggregation task")
        redis_queue.enqueue(agg_spec)

        # Wait for aggregation
        for _ in range(30):
            time.sleep(1)
            agg_result = exec_context.get_result("aggregate_results")
            if agg_result is not None:
                print("✅ aggregate_results completed")
                break

        # Step 4: Display results
        print("\nStep 4: Results")

        agg_result = exec_context.get_result("aggregate_results")
        if agg_result:
            for source_data in agg_result["sources"]:
                print(f"Source {source_data['source'].split('_')[1]}: {source_data['records']} records")
            print(f"Total aggregated: {agg_result['total_records']} records")
        else:
            print("⚠️  Aggregation not completed")

    # Summary
    print("\n=== Summary ===")
    print("✅ Distributed workflow completed")
    print("✅ 3 extraction tasks processed in parallel")
    print("✅ 1 aggregation task processed results")
    print("✅ All tasks distributed across workers")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Distributed Workflow Pattern**
#    (task1 | task2 | task3) >> aggregation
#
#    - Parallel tasks executed by different workers
#    - Results stored in Redis
#    - Aggregation task waits for all inputs
#
# 2. **Task Distribution**
#    - Tasks enqueued to Redis
#    - Workers poll and execute
#    - No coordination needed between workers
#    - Redis provides FIFO queue semantics
#
# 3. **Result Sharing**
#    - Results stored in Redis channels
#    - Available to all workers
#    - Persistent across worker restarts
#    - Accessible from main process
#
# 4. **Monitoring**
#    - Poll get_result() for completion
#    - Check queue size for pending tasks
#    - Monitor worker metrics
#    - Implement timeouts
#
# 5. **Scaling**
#    ✅ Add more workers to increase throughput
#    ✅ Workers can run on different machines
#    ✅ No code changes needed
#    ✅ Linear scaling for independent tasks
#
# 6. **Fault Tolerance**
#    - Tasks remain in Redis if worker crashes
#    - Other workers pick up tasks
#    - Results persist in Redis
#    - Implement retry logic in tasks
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Data Pipeline**:
# Extract from multiple sources in parallel, aggregate, transform, load
#
# **ML Training**:
# Train multiple models in parallel, ensemble results
#
# **Web Scraping**:
# Scrape multiple sites in parallel, aggregate data
#
# **Batch Processing**:
# Process file chunks in parallel, merge results
#
# **API Integration**:
# Call multiple APIs in parallel, combine responses
#
# ============================================================================
