"""
Redis Worker Example
=====================

This example demonstrates how to create and manage worker processes that
execute tasks from a Redis queue. Workers enable distributed execution by:
- Polling Redis queue for tasks
- Executing tasks independently
- Storing results back to Redis
- Handling errors gracefully

Prerequisites:
--------------
1. Redis server running (see redis_basics.py)
2. pip install redis

Concepts Covered:
-----------------
1. Worker process creation
2. Worker lifecycle management
3. Task polling and execution
4. Graceful shutdown
5. Worker metrics and monitoring

How to Run:
-----------
# Terminal 1: Start Redis
docker run -p 6379:6379 redis:7.2

# Terminal 2: Start worker using CLI (prefix must match demo)
python -m graflow.worker.main --worker-id worker-1 --redis-key-prefix graflow:worker_demo

# Terminal 3: Run this example to demonstrate worker usage
python examples/05_distributed/redis_worker.py

Expected Output:
----------------
=== Redis Worker Demo ===

Step 1: Starting programmatic worker
✅ Worker created: demo_worker
   Max concurrent tasks: 2
   Poll interval: 0.5s

Step 2: Enqueuing test tasks
✅ Enqueued task: task_1
✅ Enqueued task: task_2
✅ Enqueued task: task_3

Step 3: Worker processing tasks
✅ Worker started
   Worker is processing tasks...
   (Wait for workers to process tasks in background)

Step 4: Checking results after 5 seconds
✅ Task task_1: result_1
✅ Task task_2: result_2
✅ Task task_3: result_3

Step 5: Stopping worker
✅ Worker stopped gracefully
   Tasks processed: 3
   Tasks succeeded: 3
   Tasks failed: 0

=== Summary ===
✅ Worker lifecycle demonstrated
✅ Tasks executed in background
✅ Results stored in Redis
✅ Graceful shutdown working

To run workers as separate processes:
  python -m graflow.worker.main --worker-id worker-1 --redis-host localhost
"""

import sys
import threading
import time


def check_redis():
    """Check if Redis is available."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        print(f"❌ Redis not available: {e}")
        print("\nStart Redis with: docker run -p 6379:6379 redis:7.2")
        return None


def demonstrate_cli_worker():
    """Demonstrate CLI worker usage."""
    print("\n" + "="*60)
    print("CLI Worker Usage")
    print("="*60)

    print("\nTo start a worker as a separate process:")
    print("\n  python -m graflow.worker.main --worker-id worker-1 --redis-key-prefix graflow:worker_demo")
    print("\nWith custom configuration:")
    print("\n  python -m graflow.worker.main \\")
    print("    --worker-id worker-1 \\")
    print("    --redis-host localhost \\")
    print("    --redis-port 6379 \\")
    print("    --redis-key-prefix graflow:worker_demo \\")
    print("    --max-concurrent-tasks 4 \\")
    print("    --poll-interval 0.1")

    print("\nMultiple workers:")
    print("\n  # Terminal 1")
    print("  python -m graflow.worker.main --worker-id worker-1 --redis-key-prefix graflow:worker_demo")
    print("\n  # Terminal 2")
    print("  python -m graflow.worker.main --worker-id worker-2 --redis-key-prefix graflow:worker_demo")
    print("\n  # Terminal 3")
    print("  python -m graflow.worker.main --worker-id worker-3 --redis-key-prefix graflow:worker_demo")


def demonstrate_programmatic_worker(redis_client):
    """Demonstrate creating and managing a worker programmatically."""
    print("=== Redis Worker Demo ===\n")
    print("Step 1: Starting programmatic worker")

    from graflow.core.context import ExecutionContext
    from graflow.core.decorators import task
    from graflow.core.graph import TaskGraph
    from graflow.queue.base import TaskSpec
    from graflow.queue.redis import RedisTaskQueue
    from graflow.worker.worker import TaskWorker

    # Build task graph
    graph = TaskGraph()
    registered_tasks = []

    # Step 2: Create and enqueue test tasks
    print("\nStep 2: Enqueuing test tasks")

    # Create a simple workflow for testing
    @task
    def test_task_1():
        time.sleep(0.5)  # Simulate work
        return "result_1"

    @task
    def test_task_2():
        time.sleep(0.5)
        return "result_2"

    @task
    def test_task_3():
        time.sleep(0.5)
        return "result_3"

    for task_id, task in [
        ("task_1", test_task_1),
        ("task_2", test_task_2),
        ("task_3", test_task_3)
    ]:
        graph.add_node(task, task_id)
        registered_tasks.append(task)

    # Create execution context backed by Redis
    context = ExecutionContext.create(
        graph,
        "task_1",
        channel_backend="redis",
        config={"redis_client": redis_client, "key_prefix": "graflow:worker_demo"}
    )

    queue = RedisTaskQueue(
        redis_client=redis_client,
        key_prefix="graflow:worker_demo"
    )
    queue.cleanup()
    context.channel.clear()

    # Tasks are stored in Graph, no need to register separately
    # Workers will retrieve tasks from Graph via GraphStore

    # Create worker using the Redis-backed queue
    worker = TaskWorker(
        queue=queue,
        worker_id="demo_worker",
        max_concurrent_tasks=2,
        poll_interval=0.5  # Poll every 0.5 seconds
    )

    print("✅ Worker created: demo_worker")
    print("   Max concurrent tasks: 2")
    print("   Poll interval: 0.5s")

    # Enqueue tasks manually using queue TaskSpec
    # Tasks are retrieved from Graph (via GraphStore), no serialization strategy needed
    for task_id in ["task_1", "task_2", "task_3"]:
        task_node = graph.get_node(task_id)
        task_spec = TaskSpec(
            executable=task_node,
            execution_context=context
        )
        queue.enqueue(task_spec)
        print(f"✅ Enqueued task: {task_id}")

    # Step 3: Start worker in background
    print("\nStep 3: Worker processing tasks")

    worker_thread = threading.Thread(target=worker.start, daemon=True)
    worker_thread.start()

    print("✅ Worker started")
    print("   Worker is processing tasks...")
    print("   (Wait for workers to process tasks in background)")

    # Step 4: Wait for tasks to complete
    time.sleep(5)  # Give workers time to process

    print("\nStep 4: Checking results after 5 seconds")

    # Check results
    for task_id in ["task_1", "task_2", "task_3"]:
        result = context.get_result(task_id)
        if result:
            print(f"✅ Task {task_id}: {result}")
        else:
            print(f"⏳ Task {task_id}: still processing or failed")

    # Step 5: Stop worker
    print("\nStep 5: Stopping worker")
    worker.stop()
    worker_thread.join(timeout=5)

    print("✅ Worker stopped gracefully")
    print(f"   Tasks processed: {worker.tasks_processed}")
    print(f"   Tasks succeeded: {worker.tasks_succeeded}")
    print(f"   Tasks failed: {worker.tasks_failed}")


def main():
    """Run worker demonstration."""
    # Check Redis
    redis_client = check_redis()
    if not redis_client:
        sys.exit(1)

    # Demonstrate programmatic worker
    demonstrate_programmatic_worker(redis_client)

    # Show CLI usage
    demonstrate_cli_worker()

    # Summary
    print("\n=== Summary ===")
    print("✅ Worker lifecycle demonstrated")
    print("✅ Tasks executed in background")
    print("✅ Results stored in Redis")
    print("✅ Graceful shutdown working")
    print("\nTo run workers as separate processes:")
    print("  python -m graflow.worker.main --worker-id worker-1 --redis-host localhost")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Worker Creation**
#    from graflow.worker.worker import TaskWorker
#
#    worker = TaskWorker(
#        queue=redis_queue,
#        worker_id="worker-1",
#        max_concurrent_tasks=4,
#        poll_interval=0.1
#    )
#
#    - Requires a TaskQueue instance
#    - worker_id should be unique
#    - max_concurrent_tasks controls parallelism
#    - poll_interval affects latency vs CPU usage
#
# 2. **Worker Lifecycle**
#    worker.start()  # Blocks until stopped
#    worker.stop()   # Graceful shutdown
#    worker.is_running  # Check if running
#
#    - start() is blocking - run in thread for background
#    - stop() waits for current tasks to finish
#    - Use threading.Thread for programmatic workers
#
# 3. **CLI Worker (Recommended)**
#    python -m graflow.worker.main --worker-id worker-1
#
#    - Simplest way to run workers
#    - Handles signal handling automatically
#    - Supports environment variables
#    - Production-ready
#
# 4. **Worker Metrics**
#    worker.tasks_processed  # Total tasks
#    worker.tasks_succeeded  # Successful tasks
#    worker.tasks_failed     # Failed tasks
#    worker.total_execution_time  # Total time
#
#    - Access metrics for monitoring
#    - Track worker performance
#    - Implement health checks
#
# 5. **Multiple Workers**
#    # Start multiple workers in separate processes
#    # They automatically share the Redis queue
#    # Tasks are distributed among workers
#
#    - Add workers to scale horizontally
#    - Workers don't need to know about each other
#    - Redis handles task distribution
#
# 6. **Graceful Shutdown**
#    # Worker handles SIGTERM and SIGINT
#    # Finishes current tasks before stopping
#    # Timeout configurable
#
#    - Important for production deployments
#    - Prevents task loss
#    - Clean resource cleanup
#
# 7. **Fault Tolerance**
#    - If worker crashes, tasks remain in Redis
#    - Other workers can pick up tasks
#    - Implement retry logic in tasks
#    - Monitor worker health
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Start multiple workers:
#    # Terminal 1
#    python -m graflow.worker.main --worker-id worker-1
#    # Terminal 2
#    python -m graflow.worker.main --worker-id worker-2
#
# 2. Adjust concurrency:
#    python -m graflow.worker.main --max-concurrent-tasks 8
#
# 3. Use environment variables:
#    export WORKER_ID=my-worker
#    export MAX_CONCURRENT_TASKS=4
#    python -m graflow.worker.main
#
# 4. Monitor with different log levels:
#    python -m graflow.worker.main --log-level DEBUG
#
# 5. Test graceful shutdown:
#    # Start worker and send SIGINT (Ctrl+C)
#    # Watch it finish current tasks
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Multi-Server Deployment**:
# Deploy workers on multiple servers, all reading from same Redis
#
# **Auto-Scaling**:
# Monitor queue size, start/stop workers dynamically
#
# **Specialized Workers**:
# Some workers handle CPU tasks, others handle I/O tasks
#
# **Geographic Distribution**:
# Workers in different regions for low-latency processing
#
# **Resource-Specific Workers**:
# GPU workers for ML tasks, regular workers for other tasks
#
# ============================================================================
# Production Deployment:
# ============================================================================
#
# **Systemd Service**:
# [Unit]
# Description=Graflow Worker
#
# [Service]
# ExecStart=/usr/bin/python3 -m graflow.worker.main --worker-id worker-1
# Restart=always
#
# [Install]
# WantedBy=multi-user.target
#
# **Docker Container**:
# FROM python:3.11
# RUN pip install graflow redis
# CMD ["python", "-m", "graflow.worker.main", "--worker-id", "worker-1"]
#
# **Kubernetes Deployment**:
# apiVersion: apps/v1
# kind: Deployment
# metadata:
#   name: graflow-workers
# spec:
#   replicas: 3
#   template:
#     spec:
#       containers:
#       - name: worker
#         image: graflow-worker:latest
#         env:
#         - name: REDIS_HOST
#           value: redis-service
#
# ============================================================================
