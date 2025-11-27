"""
Distributed Workflow Example
==============================

This example demonstrates a complete distributed ETL workflow executed across
multiple workers using Redis as the coordination backend.

Prerequisites:
--------------
1. Redis running (auto-starts via Docker if available):
   docker run -p 6379:6379 redis:7.2
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

import atexit
import logging
import os
import random
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

def start_workers(redis_client:redis.Redis, num_workers:int = 2):
    # Start local worker threads so the example is self-contained
    # In a real deployment, these would be separate processes/containers
    redis_queue = DistributedTaskQueue(
        redis_client=redis_client,
        key_prefix="graflow:distributed_demo"
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

def main():
    """Run distributed workflow demonstration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print("=== Distributed ETL Workflow ===\n")

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

            for name, result in [
                ("extract_source_1", result1),
                ("extract_source_2", result2),
                ("extract_source_3", result3),
            ]:
                if not isinstance(result, dict) or "records" not in result:
                    raise ValueError(f"Missing or invalid result for {name}: {result}")

            # Aggregate
            total = result1["records"] + result2["records"] + result3["records"]

            return {
                "sources": [result1, result2, result3],
                "total_records": total
            }

    # Define parallel extraction followed by aggregation
        parallel_extract = (extract_source_1 | extract_source_2 | extract_source_3).set_group_name("parallel_extract").with_execution(backend=CoordinationBackend.REDIS)
        parallel_extract >> aggregate_results # type: ignore
        # Step 2: Submit workflow
        print("\nStep 2: Executing workflow with Redis backend")


        # Create execution context with Redis
        # Note: We use the same key_prefix as the workers to ensure they share the queue
        exec_context = ExecutionContext.create(
            ctx.graph,
            start_node="parallel_extract",
            channel_backend="redis",
            max_steps=100,
            config={
                "redis_client": redis_client,
                "key_prefix": "graflow:distributed_demo"
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
                agg_result = result
                for source_data in agg_result["sources"]:
                    print(f"Source {source_data['source'].split('_')[1]}: {source_data['records']} records")
                print(f"Total aggregated: {agg_result['total_records']} records")
                print("✅ Distributed workflow completed")
            else:
                print("⚠️ Distributed workflow did not return a result")

        except Exception as e:
            print(f"\n❌ Workflow execution failed: {e}")
            import traceback
            traceback.print_exc()


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
