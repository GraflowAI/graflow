# 05 - Distributed Execution

**Difficulty**: Advanced
**Status**: ✅ Complete
**Prerequisites**: Complete [04_execution](../04_execution/) first

## Overview

This section demonstrates **distributed execution** in Graflow - how to scale workflows across multiple workers using Redis as a coordination backend. Distributed execution allows you to:
- Process tasks across multiple machines
- Scale horizontally by adding workers
- Handle long-running workflows
- Implement reliable task processing

## What You'll Learn

- 🔴 Redis-based task queues
- 👷 Worker processes and task execution
- 🌐 Distributed workflow coordination
- 📡 Redis channels for inter-process communication
- ⚙️ Worker pool management

## Examples

###  1. redis_worker.py

**Concept**: Worker process setup

Learn how to create and manage worker processes that execute tasks from Redis queues.

```bash
uv run python examples/05_distributed/redis_worker.py
```

**Key Concepts**:
- TaskWorker initialization
- Worker lifecycle management
- Task dequeuing and execution
- Worker graceful shutdown

---

### 2. distributed_workflow.py

**Concept**: Complete distributed workflow

Build a complete ETL workflow that executes across multiple workers.

```bash
uv run python examples/05_distributed/distributed_workflow.py
```

**Key Concepts**:
- Distributed workflow setup
- Multiple worker coordination
- Task result sharing via Redis
- Complete pipeline execution

---

## Architecture

### Distributed Execution Model

```
┌─────────────┐
│   Main      │  1. Enqueues tasks
│  Process    │  2. Monitors completion
└──────┬──────┘
       │
       │ Redis Queue
       ▼
┌──────────────────────────────┐
│         Redis Server         │
│  - Task Queue (FIFO)         │
│  - Result Channel            │
│  - Coordination State        │
└──────────────────────────────┘
       │         ▲
       │         │
       ▼         │
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Worker  │  │ Worker  │  │ Worker  │
│   1     │  │   2     │  │   3     │
└─────────┘  └─────────┘  └─────────┘
  Dequeue     Execute      Store
  tasks       tasks        results
```

### Components

**Main Process**:
- Creates ExecutionContext with Redis backend
- Enqueues tasks to Redis queue
- Monitors workflow execution
- Retrieves final results

**Redis Server**:
- Stores task queue (FIFO order)
- Stores execution results
- Coordinates worker communication
- Provides atomic operations

**Worker Processes**:
- Poll Redis queue for tasks
- Execute task functions
- Store results back to Redis
- Handle errors gracefully

## Redis Configuration

### Task Queue Configuration

```python
from graflow.core.context import ExecutionContext
from graflow.queue.factory import QueueBackend
import redis

# Create Redis client
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    decode_responses=True
)

# Create context with Redis backend
context = ExecutionContext.create(
    graph=graph,
    start_node="start",
    queue_backend=QueueBackend.REDIS,
    channel_backend="redis",
    config={"redis_client": redis_client}
)
```

### Worker Configuration

```python
from graflow.worker.worker import TaskWorker

# Create worker
worker = TaskWorker(
    redis_host='localhost',
    redis_port=6379,
    worker_id="worker-1"
)

# Start processing
worker.start()  # Blocks until stopped
```

## Running Distributed Workflows

### Step 1: Start Redis

Using Docker:
```bash
docker run -p 6379:6379 redis:7.2
```

Using Homebrew (macOS):
```bash
brew services start redis
```

Using apt (Ubuntu/Debian):
```bash
sudo service redis-server start
```

### Step 2: Start Workers

In separate terminals:
```bash
# Terminal 1
uv run python -m graflow.worker.main --worker-id worker-1

# Terminal 2
uv run python -m graflow.worker.main --worker-id worker-2

# Terminal 3
uv run python -m graflow.worker.main --worker-id worker-3
```

### Step 3: Run Workflow

```bash
uv run python examples/05_distributed/distributed_workflow.py
```

## Distributed Patterns

### Pattern 1: Fan-Out Processing

```python
with workflow("fanout") as ctx:
    @task
    def split_data():
        return {"batches": [1, 2, 3, 4, 5]}

    @task
    def process_batch(batch_id: int):
        # Each batch processed by different worker
        return f"batch_{batch_id}_processed"

    # Create parallel tasks
    split_data >> (process_batch | process_batch | process_batch)
```

### Pattern 2: Map-Reduce

```python
with workflow("mapreduce") as ctx:
    @task
    def map_task(data: list):
        # Distribute processing
        return [process(item) for item in data]

    @task
    def reduce_task(results: list):
        # Aggregate results
        return sum(results)

    map_task >> reduce_task
```

### Pattern 3: Pipeline with Workers

```python
with workflow("pipeline") as ctx:
    @task
    def extract():
        return {"data": "raw"}

    @task
    def transform(data: dict):
        return {"data": "transformed"}

    @task
    def load(data: dict):
        return "loaded"

    extract >> transform >> load
    # Each task can execute on different worker
```

## Benefits of Distributed Execution

### Scalability
- ✅ Add more workers to handle increased load
- ✅ Process tasks in parallel across machines
- ✅ No single point of bottleneck

### Reliability
- ✅ Workers can fail and restart
- ✅ Tasks can be retried on failure
- ✅ Redis provides persistent storage

### Flexibility
- ✅ Workers can run on different machines
- ✅ Scale workers independently
- ✅ Mix local and remote execution

## Performance Considerations

### Network Latency
- Redis communication adds overhead (~1-5ms per operation)
- Use local Redis for development
- Use remote Redis for production

### Task Granularity
- **Too small**: Network overhead dominates
- **Too large**: Poor parallelization
- **Just right**: Balance network vs computation

### Worker Count
- **Too few**: Underutilized capacity
- **Too many**: Redis contention
- **Optimal**: Match available CPU cores

## Best Practices

### ✅ DO

1. **Use appropriate task granularity**
   ```python
   # Good: Meaningful work unit
   @task
   def process_batch(items: list):
       return [heavy_computation(item) for item in items]
   ```

2. **Handle worker failures**
   ```python
   # Implement retry logic
   @task
   def reliable_task():
       try:
           return risky_operation()
       except Exception as e:
           log_error(e)
           raise  # Will be retried
   ```

3. **Monitor worker health**
   ```python
   # Use worker heartbeats
   worker = TaskWorker(
       redis_host='localhost',
       heartbeat_interval=5  # seconds
   )
   ```

### ❌ DON'T

1. **Don't create too many small tasks**
   ```python
   # Bad: Too much overhead
   for i in range(10000):
       tiny_task(i)  # Each task ~1-5ms overhead
   ```

2. **Don't assume task execution order**
   ```python
   # Bad: Race condition
   task1()  # Executed by worker-1
   task2()  # Might execute before task1 completes
   # Use >> operator for dependencies
   ```

3. **Don't store large objects in Redis**
   ```python
   # Bad: 100MB dataset
   return huge_dataset

   # Good: Store reference
   save_to_s3(huge_dataset, "key")
   return {"s3_key": "key"}
   ```

## Troubleshooting

### Redis Connection Errors

```
Error: Error connecting to Redis at localhost:6379
```

**Solution**: Ensure Redis is running
```bash
# Check if Redis is running
redis-cli ping  # Should return PONG

# Start Redis if not running
docker run -p 6379:6379 redis:7.2
```

### Workers Not Processing Tasks

```
Workers idle, tasks in queue
```

**Solution**: Check worker registration
```bash
# Verify workers are registered
redis-cli SMEMBERS graflow:workers
```

### Task Serialization Errors

```
Error: cannot pickle 'module' object
```

**Solution**: Use cloudpickle-compatible code
```python
# Import inside task, not at module level
@task
def my_task():
    import numpy as np  # Inside task
    return np.array([1, 2, 3])
```

### Redis Memory Issues

```
Error: OOM command not allowed when used memory > 'maxmemory'
```

**Solution**: Configure Redis maxmemory policy
```bash
# In redis.conf
maxmemory 1gb
maxmemory-policy allkeys-lru
```

## Advanced Topics

### Custom Task Routing

Route specific tasks to specific workers:

```python
@task(handler="redis", worker_pool="gpu")
def gpu_task():
    # Only executed by GPU workers
    pass
```

### Priority Queues

Implement priority-based task execution:

```python
# High priority tasks
queue.enqueue(task, priority=10)

# Low priority tasks
queue.enqueue(task, priority=1)
```

### Result Expiration

Set TTL for task results:

```python
context.set_result(
    task_id,
    result,
    ttl=3600  # 1 hour
)
```

## Monitoring

### Redis Monitoring

```bash
# Monitor Redis operations
redis-cli MONITOR

# Check queue size
redis-cli LLEN graflow:queue

# Check worker count
redis-cli SCARD graflow:workers
```

### Worker Metrics

```python
# Access worker metrics
worker.get_metrics()
# Returns: {
#   "tasks_processed": 100,
#   "tasks_failed": 2,
#   "uptime_seconds": 3600
# }
```

## Next Steps

After mastering distributed execution:

1. **06_advanced**: Learn cycles, dynamic tasks, error handling
2. **07_real_world**: Apply concepts to real-world problems

## API Reference

**ExecutionContext**:
- `ExecutionContext.create(..., queue_backend=QueueBackend.REDIS)`
- `ExecutionContext.create(..., channel_backend="redis")`

**TaskWorker**:
- `TaskWorker(redis_host, redis_port, worker_id)`
- `worker.start()` - Start processing tasks
- `worker.stop()` - Graceful shutdown

**Redis Configuration**:
- `config={"redis_client": redis.Redis(...)}`

---

**Ready to scale workflows? Start with `redis_basics.py`! 🚀**
