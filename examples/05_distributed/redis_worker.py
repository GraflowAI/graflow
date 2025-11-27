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
1. Redis running (auto-starts via Docker if available):
   docker run -p 6379:6379 redis:7.2
2. pip install redis

Concepts Covered:
-----------------
1. Worker process creation
2. Worker lifecycle management
3. Task polling and execution
4. Graceful shutdown
5. Worker metrics and monitoring
6. Distributed workflow execution with engine.execute()

How to Run:
-----------
# Option 1: Run with embedded workers (self-contained)
python examples/05_distributed/redis_worker.py

# Option 2: Run with external workers
# Terminal 1: Start Redis
docker run -p 6379:6379 redis:7.2

# Terminal 2: Start worker using CLI (prefix must match demo)
python -m graflow.worker.main --worker-id worker-1 --redis-key-prefix graflow:worker_demo

# Terminal 3: Run this example
python examples/05_distributed/redis_worker.py

Expected Output:
----------------
=== Redis Worker Demo ===

Step 1: Setup
✅ Redis connected
Using Redis at localhost:6379
✅ 2 local worker threads started

Step 2: Executing workflow with Redis backend
⏳ Waiting for workflow completion...

Step 3: Results
✅ Task test_task_1: result_1
✅ Task test_task_2: result_2
✅ Task test_task_3: result_3
✅ Distributed workflow completed

=== Summary ===
✅ Worker lifecycle demonstrated
✅ Tasks executed by workers
✅ Results stored in Redis
✅ Graceful shutdown working

To run workers as separate processes:
  python -m graflow.worker.main --worker-id worker-1 --redis-host localhost
"""

import atexit
import logging
import os
import signal
import socket
import sys
import time

import redis

from graflow.coordination.coordinator import CoordinationBackend
from graflow.core.context import ExecutionContext
from graflow.core.decorators import task
from graflow.core.workflow import workflow
from graflow.queue.distributed import DistributedTaskQueue
from graflow.worker.worker import TaskWorker

REDIS_IMAGE = "redis:7.2"

def _is_port_available(port: int) -> bool:
    """Check if a local TCP port is free to bind."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _get_container_host_port(container):
    """Return the host port mapped to container's 6379/tcp."""
    try:
        container.reload()
        ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
        mapping = ports.get("6379/tcp")
        if mapping:
            return int(mapping[0]["HostPort"])
    except Exception as exc:  # pragma: no cover - best effort inspection
        print(f"⚠️  Could not determine Redis port: {exc}")
    return None


def _register_container_cleanup(container):
    """Register cleanup for a Docker container."""

    def _cleanup():
        try:
            container.stop()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"⚠️  Failed to stop Redis container: {exc}")

    atexit.register(_cleanup)


def _stop_redis_container(container):
    """Stop the Docker Redis container if we started one."""
    try:
        container.stop()
    except Exception as exc:  # pragma: no cover - best effort cleanup
        print(f"⚠️  Failed to stop Redis container: {exc}")


def _start_redis_container():
    """Start Redis via Docker using the redis:7.2 image on port 6379."""
    try:
        import docker  # type: ignore
    except ImportError:
        print("⚠️  Docker SDK not installed; cannot auto-start Redis container.")
        return None

    try:
        client = docker.from_env()
        container = client.containers.run(
            REDIS_IMAGE,
            ports={'6379/tcp': 6379},
            detach=True,
            remove=True
        )
        _register_container_cleanup(container)
        host_port = _get_container_host_port(container)
        if host_port is None or host_port != 6379:
            print(f"❌ Redis container did not bind to port 6379 (bound to {host_port}).")
            _stop_redis_container(container)
            return None
        return container, host_port
    except Exception as exc:  # pragma: no cover - docker import/env errors
        print(f"❌ Failed to start Redis via Docker: {exc}")
        return None


def _connect_redis(host: str = "localhost", port: int = 6379, password: str | None = None) -> redis.Redis:
    client = redis.Redis(host=host, port=port, password=password, decode_responses=True)
    client.ping()
    return client


def check_redis() -> redis.Redis | None:
    """Ensure Redis is available, starting a Docker container if needed."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD")
    try:
        return _connect_redis(host=host, port=port, password=password)
    except ImportError as exc:
        print(f"❌ redis package not installed: {exc}")
        return None
    except Exception as exc:
        print(f"⚠️  Redis not available locally: {exc}")
        message = str(exc).upper()
        if not _is_port_available(port):
            print("⚠️  Port 6379 is already in use; using the existing Redis instance.")
            if "NOAUTH" in message and not password:
                print("⚠️  Existing Redis requires authentication. Set REDIS_PASSWORD and rerun.")
                return None
            try:
                return _connect_redis(host=host, port=port, password=password)
            except Exception as retry_exc:
                print(f"❌ Failed to use existing Redis on {host}:{port}: {retry_exc}")
                return None
        if "NOAUTH" in message:
            print("⚠️  Redis at localhost:6379 requires authentication; this example expects no auth.")
            print("ℹ️  Set REDIS_PASSWORD for the example to use your existing Redis.")
            return None
        print("⏳ Attempting to start Redis via Docker (no auth) on port 6379...")

    container_info = _start_redis_container()
    if not container_info:
        print(f"\n⚠️  Start Redis manually: docker run -p 6379:6379 {REDIS_IMAGE}")
        return None
    container, host_port = container_info

    last_error = None
    for _ in range(3):
        time.sleep(2)  # Give the container time to become ready
        try:
            client = _connect_redis(port=host_port)
            if host_port == 6379:
                print("✅ Started Redis via Docker")
            else:
                print(f"✅ Started Redis via Docker on port {host_port}")
            return client
        except Exception as exc:
            last_error = exc

    print(f"❌ Redis still unavailable after starting container: {last_error}")
    _stop_redis_container(container)
    return None


def start_workers(redis_client: redis.Redis, num_workers: int = 2):
    """Start local worker threads so the example is self-contained."""
    # In a real deployment, these would be separate processes/containers
    redis_queue = DistributedTaskQueue(
        redis_client=redis_client,
        key_prefix="graflow:worker_demo"
    )
    redis_queue.cleanup()

    workers = [
        TaskWorker(queue=redis_queue, worker_id=f"worker-{i}")
        for i in range(num_workers)
    ]

    def shutdown_workers():
        print("\nStopping workers...")
        for worker in workers:
            worker.stop()
        print("✅ Workers stopped")

    atexit.register(shutdown_workers)

    # Override TaskWorker's signal handlers to ensure we exit and trigger atexit
    def handle_signal(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    for worker in workers:
        worker.start()

    print(f"✅ {num_workers} local worker threads started")
    return workers


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


def main():
    """Run distributed worker demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=== Redis Worker Demo ===\n")

    # Step 1: Setup
    print("Step 1: Setup")
    redis_client = check_redis()
    if not redis_client:
        sys.exit(1)

    print("✅ Redis connected")
    redis_host = redis_client.connection_pool.connection_kwargs.get("host", "localhost")
    redis_port = int(redis_client.connection_pool.connection_kwargs.get("port", 6379))
    print(f"Using Redis at {redis_host}:{redis_port}")

    start_workers(redis_client=redis_client, num_workers=2)

    # Create workflow
    with workflow("worker_demo") as ctx:

        @task
        def test_task_1():
            """Test task 1."""
            time.sleep(0.5)  # Simulate work
            return "result_1"

        @task
        def test_task_2():
            """Test task 2."""
            time.sleep(0.5)
            return "result_2"

        @task
        def test_task_3():
            """Test task 3."""
            time.sleep(0.5)
            return "result_3"

        # Define parallel execution with Redis backend
        parallel_tasks = (test_task_1 | test_task_2 | test_task_3).set_group_name("parallel_tasks").with_execution(backend=CoordinationBackend.REDIS)  # noqa: F841

        # Step 2: Execute workflow
        print("\nStep 2: Executing workflow with Redis backend")

        # Create execution context with Redis
        # Note: We use the same key_prefix as the workers to ensure they share the queue
        exec_context = ExecutionContext.create(
            ctx.graph,
            start_node="parallel_tasks",
            channel_backend="redis",
            max_steps=100,
            config={
                "redis_client": redis_client,
                "key_prefix": "graflow:worker_demo"
            }
        )
        exec_context.channel.clear()

        try:
            print("⏳ Waiting for workflow completion...")

            # Execute workflow using the engine
            # This will block until the workflow completes (or fails)
            # The engine handles distributed coordination via RedisCoordinator
            from graflow.core.engine import WorkflowEngine
            engine = WorkflowEngine()
            result = engine.execute(exec_context)

            print("\nStep 3: Results")
            if result:
                # Results are stored in the execution context
                for task_name in ["test_task_1", "test_task_2", "test_task_3"]:
                    task_result = exec_context.get_result(task_name)
                    if task_result:
                        print(f"✅ Task {task_name}: {task_result}")
                    else:
                        print(f"⚠️ Task {task_name}: no result")
                print("✅ Distributed workflow completed")
            else:
                print("⚠️ Distributed workflow did not return a result")

        except Exception as e:
            print(f"\n❌ Workflow execution failed: {e}")
            import traceback
            traceback.print_exc()

    # Show CLI usage
    demonstrate_cli_worker()

    # Summary
    print("\n=== Summary ===")
    print("✅ Worker lifecycle demonstrated")
    print("✅ Tasks executed by workers")
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
# 1. **Distributed Workflow Pattern**
#    with workflow("name") as ctx:
#        parallel_tasks = (task1 | task2 | task3).set_group_name("group").with_execution(backend=CoordinationBackend.REDIS)
#
#    - Use workflow context manager for graph construction
#    - Use | operator for parallel execution
#    - with_execution(backend=CoordinationBackend.REDIS) enables distributed execution
#    - Tasks are automatically distributed across workers
#
# 2. **Engine-Based Execution**
#    engine = WorkflowEngine()
#    result = engine.execute(exec_context)
#
#    - Use engine.execute() for proper workflow execution
#    - Blocks until workflow completes
#    - Handles distributed coordination automatically
#    - Results stored in exec_context
#
# 3. **Worker Creation**
#    from graflow.worker.worker import TaskWorker
#
#    worker = TaskWorker(
#        queue=redis_queue,
#        worker_id="worker-1",
#        max_concurrent_tasks=4,
#        poll_interval=0.1
#    )
#
#    - Requires a DistributedTaskQueue instance
#    - worker_id should be unique
#    - max_concurrent_tasks controls parallelism
#    - poll_interval affects latency vs CPU usage
#
# 4. **Worker Lifecycle**
#    worker.start()  # Blocks until stopped
#    worker.stop()   # Graceful shutdown
#    worker.is_running  # Check if running
#
#    - start() is blocking - run in thread for background
#    - stop() waits for current tasks to finish
#    - Use threading.Thread for programmatic workers
#
# 5. **CLI Worker (Recommended)**
#    python -m graflow.worker.main --worker-id worker-1 --redis-key-prefix graflow:worker_demo
#
#    - Simplest way to run workers
#    - Handles signal handling automatically
#    - Supports environment variables
#    - Production-ready
#
# 6. **Execution Context with Redis**
#    exec_context = ExecutionContext.create(
#        graph,
#        start_node="parallel_tasks",
#        channel_backend="redis",
#        config={"redis_client": redis_client, "key_prefix": "graflow:worker_demo"}
#    )
#
#    - channel_backend="redis" enables distributed state sharing
#    - key_prefix must match worker configuration
#    - Results accessible via exec_context.get_result(task_id)
#
# 7. **Worker Metrics**
#    worker.tasks_processed  # Total tasks
#    worker.tasks_succeeded  # Successful tasks
#    worker.tasks_failed     # Failed tasks
#    worker.total_execution_time  # Total time
#
#    - Access metrics for monitoring
#    - Track worker performance
#    - Implement health checks
#
# 8. **Multiple Workers**
#    # Start multiple workers in separate processes
#    # They automatically share the Redis queue
#    # Tasks are distributed among workers
#
#    - Add workers to scale horizontally
#    - Workers don't need to know about each other
#    - Redis handles task distribution
#
# 9. **Graceful Shutdown**
#    # Worker handles SIGTERM and SIGINT
#    # Finishes current tasks before stopping
#    # Timeout configurable
#
#    - Important for production deployments
#    - Prevents task loss
#    - Clean resource cleanup
#
# 10. **Fault Tolerance**
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
