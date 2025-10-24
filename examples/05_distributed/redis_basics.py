"""
Redis Basics Example
=====================

This example demonstrates how to set up and test Redis as a backend for
distributed Graflow execution. Redis provides:
- Distributed task queue (FIFO)
- Shared result storage
- Inter-process communication

Prerequisites:
--------------
1. Redis server running:
   - Docker: docker run -p 6379:6379 redis:7.2
   - Homebrew: brew services start redis
   - apt: sudo service redis-server start

2. Redis Python client:
   - pip install redis

Concepts Covered:
-----------------
1. Redis connection setup
2. Redis health check
3. Redis queue backend configuration
4. Redis channel backend configuration
5. Testing distributed execution readiness

Expected Output:
----------------
=== Redis Basics Demo ===

Step 1: Checking Redis availability
✅ Redis is available at localhost:6379
   Server version: 7.2.x

Step 2: Testing Redis operations
✅ Set operation successful
✅ Get operation successful
✅ Delete operation successful

Step 3: Creating ExecutionContext with Redis backend
✅ ExecutionContext created with Redis queue backend
✅ ExecutionContext created with Redis channel backend
   Session ID: 123456789
   Queue backend: redis
   Channel backend: redis

Step 4: Basic workflow with Redis
Starting execution from: test_task
✅ Task executed successfully
   Result stored in Redis: test_result

Execution completed after 1 steps

=== Summary ===
✅ Redis connection working
✅ Redis operations functional
✅ ExecutionContext with Redis backend ready
✅ Ready for distributed execution!

Next steps:
- Start worker processes (see redis_worker.py)
- Run distributed workflows (see distributed_workflow.py)
"""

import sys


def check_redis_available():
    """Check if Redis is available and return client."""
    try:
        import redis
    except ImportError:
        print("❌ Redis library not installed")
        print("Install with: pip install redis")
        return None

    try:
        client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        print(f"❌ Cannot connect to Redis: {e}")
        print("\nTo start Redis:")
        print("  Docker: docker run -p 6379:6379 redis:7.2")
        print("  Homebrew: brew services start redis")
        print("  apt: sudo service redis-server start")
        return None


def test_redis_operations(client):
    """Test basic Redis operations."""
    print("\nStep 2: Testing Redis operations")

    # Test SET
    client.set("test_key", "test_value")
    print("✅ Set operation successful")

    # Test GET
    value = client.get("test_key")
    assert value == "test_value", "Get operation failed"
    print("✅ Get operation successful")

    # Test DELETE
    client.delete("test_key")
    print("✅ Delete operation successful")


def test_redis_backend():
    """Test ExecutionContext with Redis backend."""
    print("\nStep 3: Creating ExecutionContext with Redis backend")

    import redis

    from graflow.core.context import ExecutionContext
    from graflow.core.decorators import task
    from graflow.core.graph import TaskGraph
    from graflow.queue.redis import RedisTaskQueue

    # Create Redis client
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # Create simple graph
    graph = TaskGraph()

    @task
    def test_task():
        return "test_result"

    graph.add_node(test_task, "test_task")

    # Create ExecutionContext with Redis channel backend (queue remains in-memory)
    context = ExecutionContext.create(
        graph,
        "test_task",
        channel_backend="redis",
        config={"redis_client": redis_client}
    )

    redis_queue = RedisTaskQueue(context, redis_client=redis_client)

    print("✅ ExecutionContext created (in-memory queue, Redis channel)")
    print("✅ RedisTaskQueue ready for distributed workers")
    print(f"   Session ID: {context.session_id}")
    print("   Channel backend: redis")

    return context, redis_queue


def test_basic_workflow():
    """Test basic workflow execution with Redis backend."""
    print("\nStep 4: Basic workflow with Redis")

    import redis

    from graflow.core.context import ExecutionContext
    from graflow.core.decorators import task
    from graflow.core.workflow import workflow

    # Create Redis client
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

    with workflow("redis_test") as ctx:
        @task
        def test_task():
            """Simple test task."""
            print("✅ Task executed successfully")
            return "test_result"

        # Create execution context with Redis
        exec_context = ExecutionContext.create(
            ctx.graph,
            "test_task",
            channel_backend="redis",
            config={"redis_client": redis_client}
        )

        # Execute
        from graflow.core.engine import WorkflowEngine
        engine = WorkflowEngine()
        engine.execute(exec_context)

        # Check result
        result = exec_context.get_result("test_task")
        print(f"   Result stored in Redis: {result}\n")


def main():
    """Run Redis basics demonstration."""
    print("=== Redis Basics Demo ===\n")

    # Step 1: Check Redis availability
    print("Step 1: Checking Redis availability")
    redis_client = check_redis_available()

    if redis_client is None:
        print("\n❌ Redis not available. Please start Redis and try again.")
        sys.exit(1)

    # Get server info
    try:
        info = redis_client.info()
    except Exception as exc:  # pragma: no cover - depends on Redis availability
        print(f"⚠️  Could not fetch Redis INFO: {exc}")
        version = "unknown"
    else:
        version = "unknown"
        if isinstance(info, dict):
            version = info.get("redis_version")
            if version is None:
                server_section = info.get("server")
                if isinstance(server_section, dict):
                    version = server_section.get("redis_version", "unknown")
        if not version:
            version = "unknown"

    print("✅ Redis is available at localhost:6379")
    print(f"   Server version: {version}")

    # Step 2: Test operations
    test_redis_operations(redis_client)

    # Step 3: Test Redis backend
    test_redis_backend()

    # Step 4: Test basic workflow
    test_basic_workflow()

    # Summary
    print("=== Summary ===")
    print("✅ Redis connection working")
    print("✅ Redis operations functional")
    print("✅ ExecutionContext with Redis channel ready")
    print("✅ Ready for distributed execution!")
    print("\nNext steps:")
    print("- Start worker processes (see redis_worker.py)")
    print("- Run distributed workflows (see distributed_workflow.py)")


if __name__ == "__main__":
    main()


# ============================================================================
# Key Takeaways:
# ============================================================================
#
# 1. **Redis Setup**
#    import redis
#    client = redis.Redis(host='localhost', port=6379, decode_responses=True)
#    client.ping()  # Test connection
#
#    - Use decode_responses=True for string handling
#    - Test connection before use
#    - Handle connection errors gracefully
#
# 2. **ExecutionContext with Redis**
#    context = ExecutionContext.create(
#        graph,
#        "start",
#        queue_backend=QueueBackend.REDIS,
#        channel_backend="redis",
#        config={"redis_client": redis_client}
#    )
#
#    - Pass Redis client in config
#    - Both queue and channel can use Redis
#    - Unified configuration approach
#
# 3. **Redis as Task Queue**
#    - FIFO queue for task execution
#    - Multiple workers can consume from same queue
#    - Persistent storage (survives crashes)
#    - Atomic operations
#
# 4. **Redis as Channel**
#    - Key-value store for results
#    - Shared across all workers
#    - Pub/sub capabilities
#    - TTL support for cleanup
#
# 5. **When to Use Redis Backend**
#    ✅ Multiple worker processes
#    ✅ Distributed execution across machines
#    ✅ Long-running workflows
#    ✅ Task persistence required
#    ✅ Horizontal scaling needed
#
# 6. **When NOT to Use Redis Backend**
#    ❌ Single-process workflows
#    ❌ Quick local development
#    ❌ No persistence needed
#    ❌ Low latency requirements
#    ❌ Simple sequential tasks
#
# 7. **Configuration Options**
#    redis_client = redis.Redis(
#        host='localhost',  # Redis server host
#        port=6379,         # Redis server port
#        db=0,              # Database number
#        password=None,     # Authentication password
#        decode_responses=True  # Auto-decode bytes to strings
#    )
#
# ============================================================================
# Try Experimenting:
# ============================================================================
#
# 1. Connect to remote Redis:
#    client = redis.Redis(host='redis.example.com', port=6379)
#
# 2. Use different database:
#    client = redis.Redis(host='localhost', db=1)  # Use DB 1 instead of 0
#
# 3. Set password authentication:
#    client = redis.Redis(host='localhost', password='secret')
#
# 4. Test connection with error handling:
#    try:
#        client.ping()
#        print("Connected!")
#    except redis.ConnectionError:
#        print("Connection failed")
#
# 5. Monitor Redis operations:
#    # In terminal: redis-cli MONITOR
#    # Then run this script to see all Redis commands
#
# ============================================================================
# Real-World Use Cases:
# ============================================================================
#
# **Multi-Server Deployment**:
# Deploy workers on multiple servers, all connecting to central Redis
#
# **Load Balancing**:
# Add/remove workers dynamically based on queue size
#
# **Fault Tolerance**:
# If a worker crashes, tasks remain in Redis for other workers
#
# **Horizontal Scaling**:
# Scale to hundreds of workers processing tasks in parallel
#
# **Persistent Workflows**:
# Long-running workflows survive worker restarts
#
# ============================================================================
# Troubleshooting:
# ============================================================================
#
# **Connection Refused**:
# Error: Connection refused
# Solution: Start Redis server
#
# **Authentication Failed**:
# Error: NOAUTH Authentication required
# Solution: Provide password in Redis() constructor
#
# **Wrong Database**:
# Data not visible after switch
# Solution: Ensure all clients use same db number
#
# **Out of Memory**:
# Error: OOM command not allowed
# Solution: Configure maxmemory in redis.conf
#
# **Slow Operations**:
# Operations taking too long
# Solution: Check network latency, use connection pooling
#
# ============================================================================
