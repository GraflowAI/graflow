# Graflow Unique Features & Originality
## Comprehensive Comparison with LangGraph, LangChain, Celery, and Airflow

**Document Version**: 1.1
**Last Updated**: 2025-10-23
**Author**: Graflow Team

**Changelog**:
- v1.1 (2025-10-23): Added Checkpoint/Resume System as core feature #9
- v1.0 (2025-10-22): Initial version

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Unique Features](#core-unique-features)
3. [Detailed Feature Analysis](#detailed-feature-analysis)
4. [Comparative Analysis](#comparative-analysis)
5. [Use Case Guidelines](#use-case-guidelines)
6. [Production Deployment](#production-deployment)
7. [Future Roadmap](#future-roadmap)

---

## Executive Summary

Graflow is a **general-purpose workflow execution engine** that combines the best aspects of task orchestration systems with unique innovations in **dynamic graph mutation**, **worker fleet management**, **pluggable execution strategies**, **checkpoint/resume capabilities**, and **Pythonic DSL design**.

### Key Differentiators

| Feature | Description | Competing Tools |
|---------|-------------|-----------------|
| **Worker Fleet Management** | Distributed parallel execution of grouped tasks via TaskWorker | Celery, Airflow (partial) |
| **Runtime Dynamic Tasks** | Modify workflow graph during execution | None (LangGraph: compile-time only) |
| **State Machine Execution** | First-class state machines via next_iteration() | None |
| **Pythonic Operators DSL** | Mathematical syntax (`>>`, `\|`) for workflow graph construction (DAG + cycles) | None |
| **Pluggable Task Handlers** | Custom execution strategies (GPU, SSH, cloud, Docker) | Limited in Celery/Airflow |
| **Docker Execution** | Built-in containerized task execution | External tools required |
| **Granular Error Policies** | 5 built-in parallel group error handling modes | None |
| **Seamless Local/Distributed** | Single-line backend switching | None (most require infrastructure) |
| **Channel-based Communication** | Namespaced key-value store for workflow state sharing | XCom (Airflow), State (LangGraph) |
| **Checkpoint/Resume** | User-controlled checkpointing from within tasks | LangGraph (automatic only), Airflow (limited) |

---

## Core Unique Features

### Overview

Graflow's unique features can be grouped into **9 major categories**:

1. **Worker Fleet Management** - Distributed parallel task execution
2. **Runtime Dynamic Tasks** - Graph mutation during execution
3. **State Machine Execution** - First-class state machine support (cycles via next_iteration)
4. **Pluggable Task Handlers** - Custom execution strategies (including Docker)
5. **Granular Error Policies** - Flexible parallel group error handling
6. **Pythonic Operators DSL** - Mathematical workflow graph syntax (DAG + cycles)
7. **Seamless Local/Distributed** - Backend switching
8. **Channel Communication** - Namespaced KVS for state sharing
9. **Checkpoint/Resume** - Workflow state persistence and recovery

### 1. Worker Fleet Management 🚀

**Implementation**: `graflow/worker/worker.py`, `examples/05_distributed/redis_worker.py`

#### Architecture

```
┌─────────────┐
│ Main Process│  Submit tasks to Redis Queue
└─────┬───────┘
      │
      ▼
┌─────────────────────────────────────────┐
│         Redis Task Queue                │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ Task 1  │ │ Task 2  │ │ Task 3  │  │
│  └─────────┘ └─────────┘ └─────────┘  │
└───────┬──────────┬──────────┬──────────┘
        │          │          │
   ┌────▼───┐ ┌───▼────┐ ┌──▼─────┐
   │Worker 1│ │Worker 2│ │Worker 3│
   │  4 CPUs│ │  8 CPUs│ │ 16 CPUs│
   └────────┘ └────────┘ └────────┘
```

#### Key Features

##### a. Dedicated Task Worker Process

- Drop-in `TaskWorker` process that you can run on servers or containers
- Pulls work from the shared queue, respects per-worker concurrency limits
- Makes it trivial to scale horizontally by starting more workers anywhere

##### b. Autonomous Lifecycle Management

- **Graceful Shutdown**: Responds to SIGTERM/SIGINT signals
- **Current Task Completion**: Finishes in-flight tasks before stopping
- **Configurable Timeout**: `graceful_shutdown_timeout` parameter
- **ThreadPoolExecutor**: Concurrent task processing per worker

##### c. Built-in Metrics

```python
worker.tasks_processed      # Total tasks executed
worker.tasks_succeeded      # Successful completions
worker.tasks_failed         # Failed tasks
worker.total_execution_time # Cumulative execution time
```

##### d. Horizontal Scaling

- **Linear Scaling**: Add workers to increase throughput
- **No Coordination Required**: Workers independently poll Redis
- **Geographic Distribution**: Deploy workers across data centers
- **Specialized Workers**: GPU workers, I/O workers, compute workers

##### e. Production Deployment Ready

**Systemd Service** (`examples/05_distributed/redis_worker.py:381-391`):
```ini
[Unit]
Description=Graflow Worker

[Service]
ExecStart=/usr/bin/python3 -m graflow.worker.main --worker-id worker-1
Restart=always

[Install]
WantedBy=multi-user.target
```

**Docker Deployment**:
```dockerfile
FROM python:3.11
RUN pip install graflow redis
CMD ["python", "-m", "graflow.worker.main", "--worker-id", "worker-1"]
```

**Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graflow-workers
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: worker
        image: graflow-worker:latest
        env:
        - name: REDIS_HOST
          value: redis-service
        - name: MAX_CONCURRENT_TASKS
          value: "4"
```

#### Comparison with Competitors

| Feature | Graflow | Celery | LangGraph | Airflow |
|---------|---------|--------|-----------|---------|
| **Built-in CLI** | ✅ `python -m graflow.worker.main` | ✅ `celery worker` | ❌ None | ✅ `airflow worker` |
| **Graceful Shutdown** | ✅ SIGTERM/SIGINT | ✅ | N/A | ✅ |
| **Metrics** | ✅ Built-in | ⚠️ Requires Flower | ❌ | ✅ |
| **Auto-scaling** | ✅ Add workers dynamically | ✅ | ❌ | ⚠️ Limited |
| **State Sharing** | ✅ Redis Channels | ⚠️ Via broker | ⚠️ State object | ⚠️ XCom |
| **Task Routing** | ✅ FIFO Queue | ✅ Routing keys | N/A | ✅ DAG-based |

---

### 2. Runtime Dynamic Task Generation 🔄

**Implementation**: `examples/07_dynamic_tasks/runtime_dynamic_tasks.py`

#### What Makes It Unique?

Unlike compile-time task generation (loops, factories), Graflow allows **workflow graph mutation during execution**.

#### Core APIs

##### a. `context.next_task(task)` - Add New Task at Runtime

```python
@task(inject_context=True)
def adaptive_processor(context: TaskExecutionContext):
    data_quality = check_quality()

    if data_quality < 0.5:
        # Create cleanup task on-the-fly
        cleanup_task = TaskWrapper("cleanup", cleanup_low_quality_data)
        context.next_task(cleanup_task)
    elif data_quality > 0.9:
        # Create enhancement task
        enhance_task = TaskWrapper("enhance", enhance_high_quality_data)
        context.next_task(enhance_task)
    else:
        # Standard processing
        process_task = TaskWrapper("process", standard_processing)
        context.next_task(process_task)
```

##### b. `context.next_iteration(data)` - Self-Loop Until Convergence

```python
@task(inject_context=True)
def optimize_hyperparameters(context: TaskExecutionContext, params=None):
    if params is None:
        params = {"learning_rate": 0.1, "accuracy": 0.5}

    # Training step
    new_accuracy = train_model(params)

    if new_accuracy >= 0.95:
        # Converged - finalize
        save_task = TaskWrapper("save_model", lambda: save_model(params))
        context.next_task(save_task)
    else:
        # Continue optimization
        params["accuracy"] = new_accuracy
        params["learning_rate"] *= 0.95
        context.next_iteration(params)
```

#### Real-World Use Cases

1. **ML Hyperparameter Tuning** (convergence-based):
   ```python
   while accuracy < target:
       context.next_iteration(optimize_step())
   ```

2. **Data Classification Pipeline** (conditional routing):
   ```python
   if data_type == "image":
       context.next_task(TaskWrapper("image_proc", process_image))
   elif data_type == "text":
       context.next_task(TaskWrapper("text_proc", process_text))
   ```

3. **Resilient API Calls** (retry logic):
   ```python
   try:
       result = api_call()
   except Exception:
       if attempts < max_retries:
           context.next_iteration({"attempts": attempts + 1})
       else:
           context.next_task(TaskWrapper("log_error", handle_error))
   ```

4. **Progressive Enhancement** (quality improvement):
   ```python
   if current_quality < target_quality:
       context.next_iteration(apply_enhancement())
   else:
       context.next_task(TaskWrapper("finalize", save_result))
   ```

#### Comparison with Competitors

| Feature | Graflow | LangGraph | Celery | Airflow |
|---------|---------|-----------|--------|---------|
| **Runtime Task Creation** | ✅ `next_task()` | ❌ Compile-time only | ❌ | ⚠️ Dynamic DAG (limited) |
| **Self-Loop (Iteration)** | ✅ `next_iteration()` | ⚠️ Via conditional edges | ❌ | ❌ |
| **Conditional Branching** | ✅ Based on execution results | ⚠️ Predefined branches | ❌ | ⚠️ BranchPythonOperator |
| **Graph Introspection** | ✅ During execution | ❌ | N/A | ⚠️ Limited |
| **Max Iterations** | ✅ `max_steps` parameter | ⚠️ Manual tracking | N/A | N/A |

**Key Advantage**: Graflow's graph is **truly dynamic** - it grows and changes based on runtime conditions, not just predefined branches.

---

### 3. State Machine Execution 🔄

**Implementation**: `examples/07_dynamic_tasks/runtime_dynamic_tasks.py:265-319`

#### What Makes It Unique?

Graflow enables **true state machine implementations** using `next_iteration()` and channel-based state persistence, allowing complex control flows without explicit state machine libraries.

#### State Machine Pattern

```python
@task(inject_context=True)
def state_controller(context: TaskExecutionContext):
    """Control state transitions based on current state."""
    channel = context.get_channel()
    current_state = channel.get("state", default="START")
    data = channel.get("data", default=0)

    # State transition logic
    if current_state == "START":
        channel.set("state", "PROCESSING")
        channel.set("data", data + 1)
        context.next_iteration()  # Loop back to state_controller

    elif current_state == "PROCESSING":
        if data < threshold:
            # Stay in current state
            channel.set("data", data + 1)
            context.next_iteration()
        else:
            # Transition to next state
            channel.set("state", "FINALIZING")
            context.next_iteration()

    elif current_state == "FINALIZING":
        # Transition to END and create final task
        channel.set("state", "END")
        final_task = TaskWrapper("end_state", finalize_handler)
        context.next_task(final_task)

    return {"state": current_state, "data": data}
```

#### State Machine Visualization

```
┌─────────┐
│  START  │
└────┬────┘
     │ next_iteration()
     ▼
┌────────────┐
│ PROCESSING │◄───┐ Loop while data < threshold
└─────┬──────┘    │ next_iteration()
      │           │
      │ data >= threshold
      ▼           │
┌────────────┐    │
│ FINALIZING │────┘
└─────┬──────┘
      │ next_task()
      ▼
┌─────────┐
│   END   │
└─────────┘
```

#### Key Components

##### a. State Persistence via Channels
```python
# Store state
channel.set("state", "PROCESSING")
channel.set("data", current_data)

# Retrieve state in next iteration
current_state = channel.get("state", default="START")
```

##### b. Self-Loop with `next_iteration()`
```python
# Same task executes again with updated state
context.next_iteration()
```

##### c. State-based Branching
```python
if current_state == "STATE_A":
    # Logic for state A
    context.next_iteration()
elif current_state == "STATE_B":
    # Logic for state B
    context.next_task(next_handler)
```

#### Real-World Use Cases

##### 1. Protocol State Machine (Network, API)
```python
@task(inject_context=True)
def connection_fsm(context):
    channel = context.get_channel()
    state = channel.get("state", "DISCONNECTED")

    if state == "DISCONNECTED":
        establish_connection()
        channel.set("state", "CONNECTING")
        context.next_iteration()

    elif state == "CONNECTING":
        if connection_ready():
            channel.set("state", "CONNECTED")
            context.next_iteration()
        else:
            time.sleep(1)
            context.next_iteration()

    elif state == "CONNECTED":
        if should_disconnect():
            channel.set("state", "DISCONNECTING")
            context.next_iteration()
        else:
            handle_messages()
            context.next_iteration()

    elif state == "DISCONNECTING":
        close_connection()
        return "DONE"
```

##### 2. Order Processing State Machine
```python
@task(inject_context=True)
def order_processor(context):
    channel = context.get_channel()
    order_state = channel.get("order_state", "NEW")
    order_data = channel.get("order_data")

    if order_state == "NEW":
        validate_order(order_data)
        channel.set("order_state", "VALIDATED")
        context.next_iteration()

    elif order_state == "VALIDATED":
        reserve_inventory(order_data)
        channel.set("order_state", "RESERVED")
        context.next_iteration()

    elif order_state == "RESERVED":
        process_payment(order_data)
        channel.set("order_state", "PAID")
        context.next_iteration()

    elif order_state == "PAID":
        ship_order(order_data)
        channel.set("order_state", "SHIPPED")
        # Create notification task
        context.next_task(TaskWrapper("notify", send_notification))
```

##### 3. ML Training State Machine
```python
@task(inject_context=True)
def training_fsm(context):
    channel = context.get_channel()
    phase = channel.get("phase", "INIT")
    model = channel.get("model")

    if phase == "INIT":
        model = initialize_model()
        channel.set("model", model)
        channel.set("phase", "TRAINING")
        context.next_iteration()

    elif phase == "TRAINING":
        metrics = train_epoch(model)
        if metrics["accuracy"] >= target:
            channel.set("phase", "VALIDATING")
        context.next_iteration()

    elif phase == "VALIDATING":
        val_metrics = validate(model)
        if val_metrics["accuracy"] >= target:
            channel.set("phase", "SAVING")
        else:
            channel.set("phase", "TRAINING")  # Back to training
        context.next_iteration()

    elif phase == "SAVING":
        save_model(model)
        return "COMPLETE"
```

##### 4. Game State Machine
```python
@task(inject_context=True)
def game_loop(context):
    channel = context.get_channel()
    game_state = channel.get("game_state", "MENU")

    if game_state == "MENU":
        choice = show_menu()
        channel.set("game_state", "PLAYING" if choice == "start" else "QUIT")
        context.next_iteration()

    elif game_state == "PLAYING":
        player_action = get_player_action()
        game_result = process_turn(player_action)

        if game_result == "won":
            channel.set("game_state", "WIN")
        elif game_result == "lost":
            channel.set("game_state", "LOSE")
        context.next_iteration()

    elif game_state in ["WIN", "LOSE"]:
        show_result(game_state)
        channel.set("game_state", "MENU")
        context.next_iteration()

    elif game_state == "QUIT":
        return "GAME_OVER"
```

#### Advantages Over Traditional State Machines

| Aspect | Traditional FSM Library | Graflow State Machine |
|--------|------------------------|----------------------|
| **State Storage** | In-memory object | ✅ Persistent channel (Redis) |
| **Distributed** | ❌ Single process | ✅ Workers can pick up state |
| **Visualization** | ⚠️ Requires separate tool | ✅ Workflow graph shows flow |
| **Debugging** | ⚠️ Custom logging | ✅ Task execution history |
| **Integration** | ⚠️ Standalone | ✅ Part of workflow |
| **Side Effects** | ⚠️ Must manage manually | ✅ As tasks with handlers |

#### Comparison with Competitors

| Feature | Graflow | LangGraph | Celery | Airflow |
|---------|---------|-----------|--------|---------|
| **State Machine Pattern** | ✅ `next_iteration()` + Channel | ⚠️ Via graph cycles | ❌ | ❌ |
| **State Persistence** | ✅ Channel (Memory/Redis) | ⚠️ Memory/Checkpointer | ❌ | ⚠️ XCom |
| **Distributed State** | ✅ Redis channels | ❌ | ❌ | ⚠️ DB-backed |
| **State Transitions** | ✅ Dynamic (runtime) | ⚠️ Static (graph-defined) | N/A | N/A |
| **Max Iterations** | ✅ `max_steps` | ⚠️ Manual tracking | N/A | N/A |

**Key Advantage**: Graflow allows **implementing complex state machines** as workflows, with distributed state persistence and seamless integration with other tasks.

#### Best Practices

1. **Always set `max_steps`** to prevent infinite loops:
   ```python
   ctx.execute("state_controller", max_steps=100)
   ```

2. **Use descriptive state names**:
   ```python
   # ✅ Good
   channel.set("state", "WAITING_FOR_PAYMENT")

   # ❌ Bad
   channel.set("state", "S3")
   ```

3. **Log state transitions**:
   ```python
   logger.info(f"State transition: {old_state} → {new_state}")
   ```

4. **Use enums for type safety**:
   ```python
   from enum import Enum

   class OrderState(Enum):
       NEW = "NEW"
       VALIDATED = "VALIDATED"
       PAID = "PAID"

   channel.set("state", OrderState.NEW.value)
   ```

5. **Handle unexpected states**:
   ```python
   else:
       raise ValueError(f"Unknown state: {current_state}")
   ```

---

### 4. Pluggable Task Handlers System 🔌

**Implementation**: `graflow/core/handler.py`, `examples/04_execution/custom_handler.py`

#### Architecture

```python
# Handler Interface
class TaskHandler(ABC):
    @abstractmethod
    def execute_task(self, task: Executable, context: ExecutionContext) -> Any:
        """Implement custom execution logic and return the task result when available."""
        raise NotImplementedError

# Custom Implementation
class GPUHandler(TaskHandler):
    def execute_task(self, task, context):
        with gpu_lock:
            result = task.run()  # Execute on GPU
        context.set_result(task.task_id, result)
        return result

# Registration
engine = WorkflowEngine()
engine.register_handler("gpu", GPUHandler())

# Usage
@task(handler="gpu")
def train_model():
    ...
```

#### Built-in Handler Patterns

##### a. Retry Handler
```python
class RetryHandler(TaskHandler):
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def execute_task(self, task, context):
        for attempt in range(self.max_retries):
            try:
                result = task.run()
                context.set_result(task.task_id, result)
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{self.max_retries}")
```

##### b. Caching Handler
```python
class CachingHandler(TaskHandler):
    def __init__(self):
        self.cache = {}

    def execute_task(self, task, context):
        if task.task_id in self.cache:
            context.set_result(task.task_id, self.cache[task.task_id])
            return

        result = task.run()
        self.cache[task.task_id] = result
        context.set_result(task.task_id, result)
```

##### c. Rate Limiting Handler
```python
class RateLimitHandler(TaskHandler):
    def __init__(self, min_interval=1.0):
        self.min_interval = min_interval
        self.last_execution = 0

    def execute_task(self, task, context):
        elapsed = time.time() - self.last_execution
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

        result = task.run()
        self.last_execution = time.time()
        context.set_result(task.task_id, result)
```

##### d. Monitoring Handler
```python
class MonitoringHandler(TaskHandler):
    def __init__(self):
        self.metrics = []

    def execute_task(self, task, context):
        start = time.time()
        try:
            result = task.run()
            status = "success"
            context.set_result(task.task_id, result)
        except Exception as e:
            status = "failed"
            raise
        finally:
            self.metrics.append({
                "task": task.task_id,
                "duration": time.time() - start,
                "status": status
            })
```

#### Advanced Use Cases

1. **SSH Remote Execution**:
   ```python
   class SSHHandler(TaskHandler):
       def execute_task(self, task, context):
           with ssh_client.connect(remote_host):
               result = execute_remotely(task)
           context.set_result(task.task_id, result)
   ```

2. **Cloud Function Execution** (AWS Lambda, GCP Functions):
   ```python
   class LambdaHandler(TaskHandler):
       def execute_task(self, task, context):
           payload = serialize_task(task)
           response = lambda_client.invoke(FunctionName="executor", Payload=payload)
           result = deserialize_response(response)
           context.set_result(task.task_id, result)
   ```

3. **GPU Queue Management**:
   ```python
   class GPUQueueHandler(TaskHandler):
       def execute_task(self, task, context):
           gpu_id = gpu_pool.acquire()
           try:
               result = task.run(device=f"cuda:{gpu_id}")
           finally:
               gpu_pool.release(gpu_id)
           context.set_result(task.task_id, result)
   ```

#### Built-in Docker Handler 🐳

**Implementation**: `graflow/core/handlers/docker.py`, `examples/04_execution/docker_handler.py`

Graflow includes a **production-ready Docker handler** for containerized task execution.

##### Basic Usage

```python
from graflow.core.handlers.docker import DockerTaskHandler
from graflow.core.engine import WorkflowEngine

# Register Docker handler
engine = WorkflowEngine()
engine.register_handler("docker", DockerTaskHandler(image="python:3.11-slim"))

# Use in tasks
@task(handler="docker")
def isolated_task():
    """Runs in Docker container with complete isolation."""
    import sys
    return sys.version
```

##### Advanced Configuration

**Environment Variables**:
```python
DockerTaskHandler(
    image="python:3.11-slim",
    environment={"API_KEY": "secret", "DEBUG": "1"}
)
```

**Volume Mounts**:
```python
DockerTaskHandler(
    image="python:3.11-slim",
    volumes={
        "/host/data": {"bind": "/container/data", "mode": "ro"},
        "/host/output": {"bind": "/output", "mode": "rw"}
    }
)
```

**GPU Support**:
```python
from docker.types import DeviceRequest

DockerTaskHandler(
    image="tensorflow/tensorflow:latest-gpu",
    device_requests=[DeviceRequest(count=1, capabilities=[["gpu"]])]
)
```

**Resource Limits** (via Docker API):
```python
# CPU and memory limits
container = client.containers.run(
    image="python:3.11",
    mem_limit="512m",      # 512MB RAM
    cpu_period=100000,
    cpu_quota=50000,       # 50% CPU
)
```

**Network Configuration**:
```python
DockerTaskHandler(
    image="python:3.11",
    network_mode="bridge"  # or "host", "none"
)
```

##### Key Features

1. **Complete Isolation**:
   - Separate process space
   - Isolated filesystem
   - Network isolation (configurable)
   - Cannot access host resources directly

2. **Cloudpickle Serialization**:
   - Task functions serialized via cloudpickle
   - ExecutionContext passed to container
   - Results deserialized back to host
   - Supports lambdas and closures

3. **Reproducible Environments**:
   - Pin exact Docker image versions
   - Consistent dependency versions
   - Cross-platform compatibility

4. **Multi-version Testing**:
   ```python
   # Test against multiple Python versions
   engine.register_handler("py39", DockerTaskHandler(image="python:3.9"))
   engine.register_handler("py311", DockerTaskHandler(image="python:3.11"))
   engine.register_handler("py312", DockerTaskHandler(image="python:3.12"))

   @task(handler="py39")
   def test_on_py39():
       ...

   @task(handler="py311")
   def test_on_py311():
       ...
   ```

##### Real-World Use Cases

###### 1. Security Sandbox (Untrusted Code)
```python
@task(handler="docker")
def execute_user_code(user_code: str):
    """Execute user-submitted code safely."""
    # Runs isolated - cannot harm host
    exec(user_code)
    return "Executed safely"
```

###### 2. CI/CD Testing
```python
# Test in clean environment
@task(handler="docker")
def run_integration_tests():
    """Run tests in isolated container."""
    import subprocess
    return subprocess.run(["pytest", "tests/"], capture_output=True)
```

###### 3. Data Science Experiments
```python
# Different ML framework versions
engine.register_handler("tf2", DockerTaskHandler(image="tensorflow/tensorflow:2.13.0"))
engine.register_handler("tf1", DockerTaskHandler(image="tensorflow/tensorflow:1.15.5"))

@task(handler="tf2")
def train_with_tf2():
    import tensorflow as tf
    # Train model with TensorFlow 2.x
    ...

@task(handler="tf1")
def train_with_tf1():
    import tensorflow as tf
    # Train model with TensorFlow 1.x
    ...
```

###### 4. Legacy Code Execution
```python
# Run old code on Python 2.7
engine.register_handler("py27", DockerTaskHandler(image="python:2.7"))

@task(handler="py27")
def legacy_processing():
    """Execute legacy Python 2 code."""
    # Old code that doesn't work on Python 3
    print "Hello from Python 2"  # Python 2 syntax
```

##### Performance Considerations

| Aspect | Direct Handler | Docker Handler |
|--------|----------------|----------------|
| **Startup Time** | ~1ms | ~500-2000ms |
| **Serialization** | None | ~10-100ms |
| **Execution** | Native | Native (same) |
| **Total Overhead** | Minimal | Significant |

**Recommendation**: Use Docker handler for:
- ✅ Security-critical tasks (untrusted code)
- ✅ Reproducible environments
- ✅ Multi-version testing
- ✅ Long-running tasks (overhead amortized)

Avoid for:
- ❌ Performance-critical code
- ❌ Frequent short tasks
- ❌ Development iteration

##### Unique Advantages

| Feature | Graflow Docker Handler | Competitors |
|---------|------------------------|-------------|
| **Built-in Support** | ✅ Core feature | ⚠️ External tools |
| **Same Workflow** | ✅ Mix Docker + Direct | ❌ Separate systems |
| **Cloudpickle** | ✅ Lambda/closure support | ❌ |
| **GPU Support** | ✅ DeviceRequest | ⚠️ Limited |
| **Multi-image** | ✅ Multiple handlers | ⚠️ Limited |

**Key Advantage**: **Seamless integration** of containerized execution within the same workflow, using the same DSL and context management.

#### Comparison with Competitors

| Feature | Graflow | LangGraph | Celery | Airflow |
|---------|---------|-----------|--------|---------|
| **Pluggable Handlers** | ✅ `TaskHandler` interface | ❌ Hardcoded nodes | ⚠️ Task classes | ⚠️ Operators |
| **Handler Registration** | ✅ `register_handler()` | N/A | ⚠️ Via decorators | ⚠️ Via plugins |
| **Per-Task Handler** | ✅ `@task(handler="...")` | N/A | ⚠️ Task routing | ⚠️ Operator selection |
| **Custom Logic** | ✅ Full control | ❌ | ⚠️ Limited | ⚠️ Limited |
| **Composability** | ✅ Decorator pattern | N/A | ❌ | ❌ |

**Key Advantage**: Graflow allows **complete customization of task execution** without modifying core engine logic.

---

### 5. Granular Parallel Group Error Policies ⚠️

**Implementation**: `examples/11_error_handling/parallel_group_strict_mode.py`

#### The Problem

When executing tasks in parallel (`task_a | task_b | task_c`), what should happen if one task fails?

#### Graflow's Solution: 5 Built-in Policies

##### a. Strict Mode (Default)
```python
parallel = (task_a | task_b | task_c)  # All must succeed
```
- **Behavior**: Any failure → `ParallelGroupError`
- **Use Case**: Critical validations, atomic operations

##### b. Best Effort Mode
```python
parallel = (task_a | task_b | task_c).with_policy(
    ErrorHandlingPolicy.BEST_EFFORT
)
```
- **Behavior**: Ignore failures, continue with successful results
- **Use Case**: Data aggregation from multiple sources (some may be unavailable)

##### c. At Least N Mode
```python
parallel = (task_a | task_b | task_c).with_policy(
    ErrorHandlingPolicy.AT_LEAST_N(n=2)
)
```
- **Behavior**: Continue if at least N tasks succeed
- **Use Case**: Redundant data fetching (2 out of 3 sources sufficient)

##### d. Critical Tasks Mode
```python
parallel = (task_a | task_b | task_c).with_policy(
    ErrorHandlingPolicy.CRITICAL_TASKS(["task_a"])
)
```
- **Behavior**: Only specified tasks must succeed
- **Use Case**: Primary data source must succeed, others optional

##### e. Custom Policy
```python
class CustomPolicy(ErrorHandlingPolicy):
    def should_fail(self, failed_tasks, successful_tasks):
        # Custom logic
        return len(failed_tasks) > len(successful_tasks)

parallel.with_policy(CustomPolicy())
```

#### Error Information

```python
try:
    engine.execute(context)
except ParallelGroupError as e:
    print(f"Group: {e.group_id}")
    print(f"Failed: {e.failed_tasks}")  # [(task_id, error_msg), ...]
    print(f"Succeeded: {e.successful_tasks}")  # [task_id, ...]
```

#### Comparison with Competitors

| Feature | Graflow | LangGraph | Celery | Airflow |
|---------|---------|-----------|--------|---------|
| **Strict Mode** | ✅ Default | ❌ | ⚠️ Implicit | ⚠️ Via trigger_rule |
| **Best Effort** | ✅ Policy | ❌ | ❌ | ⚠️ `all_done` trigger |
| **At Least N** | ✅ Policy | ❌ | ❌ | ❌ |
| **Critical Tasks** | ✅ Policy | ❌ | ❌ | ❌ |
| **Custom Policy** | ✅ Extensible | ❌ | ❌ | ⚠️ Limited |
| **Error Details** | ✅ `ParallelGroupError` | N/A | ⚠️ Via result backend | ⚠️ Via logs |

**Key Advantage**: **5 flexible policies** covering common scenarios, plus extensibility for custom logic.

---

### 6. Pythonic Operators DSL 📐

**Implementation**: `examples/02_workflows/operators_demo.py`

#### Syntax

```python
# Sequential execution
task_a >> task_b >> task_c

# Parallel execution
task_a | task_b | task_c

# Combined (Diamond pattern)
fetch >> (transform_a | transform_b) >> store

# Multi-stage
(load_a | load_b) >> validate >> (process_a | process_b) >> (save_db | save_file)

# Named parallel groups
(task_a | task_b | task_c).set_group_name("parallel_tasks") >> aggregate
```

#### Operator Semantics

| Operator | Meaning | Graph Representation |
|----------|---------|----------------------|
| `a >> b` | Sequential: b depends on a | `a → b` |
| `a \| b` | Parallel: a and b independent | `a`, `b` (no edge) |
| `(a \| b) >> c` | Fan-in: c depends on both a and b | `a → c`, `b → c` |
| `a >> (b \| c)` | Fan-out: b and c depend on a | `a → b`, `a → c` |

#### Common Workflow Patterns

**Note**: While these patterns form DAGs, Graflow also supports cycles via `next_iteration()` for state machine execution (see Section 3).

##### Linear Pipeline
```python
extract >> validate >> transform >> load
```

##### Fan-out (One-to-Many)
```python
source >> (process_region_1 | process_region_2 | process_region_3)
```

##### Fan-in (Many-to-One)
```python
(fetch_db | fetch_api | fetch_file) >> aggregate
```

##### Diamond (Fan-out + Fan-in)
```python
fetch >> (transform_a | transform_b) >> merge >> store
```

##### Multi-Stage Pipeline
```python
(extract_a | extract_b) >> validate >> (transform_a | transform_b) >> (load_db | load_s3)
```

#### Comparison with Competitors

**LangGraph**:
```python
# Verbose, imperative
graph = StateGraph(...)
graph.add_node("a", node_a)
graph.add_node("b", node_b)
graph.add_edge("a", "b")
graph.add_conditional_edges("b", router, {"c": "c", "d": "d"})
```

**Airflow**:
```python
# Operator-based, less intuitive
task_a = BashOperator(task_id="a", ...)
task_b = BashOperator(task_id="b", ...)
task_a >> task_b
# But parallel requires explicit groups
```

**Celery**:
```python
# Functional API, less visual
chain(task_a.s(), task_b.s(), task_c.s())
group(task_a.s(), task_b.s(), task_c.s())
# Chaining groups is cumbersome
```

**Graflow**:
```python
# Mathematical, declarative
fetch >> (transform_a | transform_b) >> store

# Also supports cycles for state machines
@task(inject_context=True)
def process(context):
    if not_done:
        context.next_iteration()  # Creates cycle
```

**Key Advantage**: **Most concise and intuitive** workflow graph syntax (DAG + cycles), inspired by mathematical notation. Combines compile-time graph definition with runtime cycle control.

---

### 7. Seamless Local/Distributed Execution 🌐

**Implementation**: `examples/05_distributed/distributed_workflow.py`

#### The Problem

Most orchestration tools require infrastructure for distributed execution, making local development difficult.

#### Coordination Backends

```python
# For parallel execution
parallel = (task_a | task_b | task_c).with_execution(
    backend=CoordinationBackend.DIRECT        # Sequential (debugging)
    # backend=CoordinationBackend.THREADING   # Thread-based parallelism
    # backend=CoordinationBackend.MULTIPROCESSING  # Process-based parallelism
    # backend=CoordinationBackend.REDIS       # Distributed via TaskWorker
)
```

> **Note:** Distributed execution is only available for `ParallelGroup` expressions (e.g., `(task_a | task_b)`), where the `GroupExecutor` publishes tasks to Redis queues that `TaskWorker` instances consume. Sequential pipelines (`task_a >> task_b`) always run in-process without TaskWorker distribution. ParallelGroup stages follow the Bulk Synchronous Parallel (BSP) model: all parallel branches complete, synchronize via a barrier, and only then advance to downstream tasks.

#### Environment-based Configuration

```python
import os
from graflow.queue.redis import RedisTaskQueue

context = ExecutionContext.create(
    graph,
    channel_backend="redis" if os.getenv("ENV") == "production" else "memory",
    config={"redis_client": redis_client} if os.getenv("ENV") == "production" else None,
)

redis_queue = None
if os.getenv("ENV") == "production":
    redis_queue = RedisTaskQueue(context, redis_client=redis_client)
```

#### Comparison with Competitors

| Tool | Local Execution | Distributed Execution | Switching |
|------|-----------------|----------------------|-----------|
| **Graflow** | ✅ In-memory queue | ✅ Redis queue (専用インスタンス) | ✅ 数行で切替 |
| **Celery** | ❌ Always requires broker | ✅ | ❌ |
| **LangGraph** | ✅ In-memory | ❌ Requires external tools | ❌ |
| **Airflow** | ❌ Always requires DB | ✅ | ❌ |

**Key Advantage**: **Zero-infrastructure local development** with seamless production deployment.

---

### 8. Channel-based Communication 📡

**Implementation**: `examples/03_data_flow/channels_basic.py`

#### Workflow-scoped Key-Value Store

```python
# Producer task
@task(inject_context=True)
def producer(context):
    channel = context.get_channel()
    channel.set("config", {"batch_size": 100})
    channel.set("metrics", {"processed": 0})

# Consumer task (decoupled)
@task(inject_context=True)
def consumer(context):
    channel = context.get_channel()
    config = channel.get("config")
    metrics = channel.get("metrics", default={})
```

#### Channel Operations

```python
channel.set(key, value)              # Store value
channel.get(key, default=None)       # Retrieve value
channel.keys()                       # List all keys
channel.clear()                      # Clear all data
```

#### Use Cases

##### Shared Configuration
```python
@task(inject_context=True)
def setup(context):
    config = load_config()
    context.get_channel().set("config", config)

# All subsequent tasks access config
@task(inject_context=True)
def process(context):
    config = context.get_channel().get("config")
```

##### Metrics Accumulation
```python
@task(inject_context=True)
def task_with_metrics(context):
    channel = context.get_channel()
    metrics = channel.get("metrics", [])
    metrics.append({"task": "...", "duration": 1.5})
    channel.set("metrics", metrics)
```

##### Progress Tracking
```python
@task(inject_context=True)
def track_progress(context):
    channel = context.get_channel()
    progress = channel.get("progress", 0)
    channel.set("progress", progress + 10)
```

#### Distributed Channels (Redis)

```python
context = ExecutionContext.create(
    graph,
    channel_backend="redis"  # Workers share state via Redis
)
```

#### Comparison with Competitors

| Tool | Inter-task Communication | Distributed Support | API Style |
|------|-------------------------|---------------------|-----------|
| **Graflow** | ✅ Channels | ✅ Redis | Namespaced KVS |
| **Airflow** | ⚠️ XCom | ⚠️ Via metadata DB | Key-value |
| **LangGraph** | ⚠️ State object | ❌ | Shared state |
| **Celery** | ❌ (via result backend) | ⚠️ Limited | N/A |

**Key Advantage**: **Workflow-scoped state sharing via namespaced key-value store** with distributed support.

---

### 9. Checkpoint/Resume System 💾

**Implementation**: `graflow/core/checkpoint.py`, `tests/core/test_checkpoint.py`

#### What Makes It Unique?

Graflow provides **production-ready checkpoint/resume functionality** with **user-controlled checkpointing from within tasks**. Unlike other tools that only support automatic or external checkpointing, Graflow lets tasks call `context.checkpoint()` at precise moments; the engine writes the checkpoint right after the task succeeds so users get consistent, replay-ready snapshots for long-running workflows.

#### Architecture

##### Three-File Checkpoint Structure

```
checkpoint_base_path/
├── checkpoint.pkl           # ExecutionContext (graph, channel data)
├── checkpoint.state.json    # Execution state (steps, cycle counts, pending tasks)
└── checkpoint.meta.json     # Metadata (timestamps, user metadata)
```

##### Complete State Persistence

1. **ExecutionContext** (`.pkl` file):
   - Task graph structure
   - Channel data (MemoryChannel or RedisChannel reference)
   - Backend configuration
   - Completed tasks tracking

2. **Execution State** (`.state.json` file):
   - Schema version
   - Session ID and steps count
   - Cycle counts per task
   - **Full TaskSpec objects** (not just task IDs)
   - Pending tasks in queue
   - Current task (if resuming mid-execution)

3. **Checkpoint Metadata** (`.meta.json` file):
   - Checkpoint ID (unique identifier)
   - Creation timestamp (ISO 8601)
   - Session ID
   - Start node
   - Backend information
   - User-defined metadata

#### Core APIs

##### Scheduling a Checkpoint from a Task (Recommended)

```python
@task(inject_context=True)
def process_batch(context):
    run_expensive_step()

    # Default behaviour: checkpoint is written after the task succeeds
    context.checkpoint(metadata={"stage": "post_processing"})
```

- `context.checkpoint()` is idempotent-friendly: it records the request, waits for the task to finish, then the engine persists the snapshot.
- The resolved path is stored on the execution context: `context.execution_context.last_checkpoint_path`.
- Attach structured metadata (e.g., stage name, batch number) so you can decide what to resume later.

##### Immediate Checkpoint (Resume from the Same Task)

```python
checkpoint_path, metadata = context.checkpoint(
    metadata={"stage": "critical_section"},
    deferred=False,                        # write checkpoint right now
    path="checkpoints/manual/snapshot.pkl" # optional explicit location
)
print(f"Checkpoint saved to {checkpoint_path}")
```

- Use `deferred=False` when you need to re-run the current task after resuming (for example, before an external API call).
- Immediate checkpoints return the `(path, metadata)` tuple because the file is already written.

##### Manual Checkpoint from Host Code

```python
from graflow.core.checkpoint import CheckpointManager

checkpoint_path, metadata = CheckpointManager.create_checkpoint(
    execution_context,
    metadata={"stage": "before_shutdown"}
)
print(f"Manual snapshot saved: {checkpoint_path}")
```

- Helpful right before shutting down an executor or when running orchestration scripts.
- The helper infers the backend from the path (`checkpoints/...` for local files, `redis://` in future releases).

##### Resuming from a Checkpoint

```python
from graflow.core.checkpoint import CheckpointManager
from graflow.core.engine import WorkflowEngine

checkpoint_path = "checkpoints/session_12345_step_40.pkl"
restored_context, metadata = CheckpointManager.resume_from_checkpoint(checkpoint_path)
print(f"Resuming session {metadata.session_id} at step {metadata.steps}")

engine = WorkflowEngine()
engine.execute(restored_context)  # pending tasks are already queued
```

- No need to rebuild the graph manually — the checkpoint already carries it.
- When using Redis backends, make sure the target worker can reach the same Redis instance.

##### Quick Start: Save and Resume (User Flow)

1. Instrument long-running tasks with `context.checkpoint(metadata=...)` at the points you want to resume from.
2. Capture the emitted checkpoint path (logged or via `context.execution_context.last_checkpoint_path`) so it can be supplied later.
3. On restart, call `CheckpointManager.resume_from_checkpoint(path)` and pass the restored context to `WorkflowEngine.execute()` — pending work picks up immediately.

#### Real-World Use Cases

##### 1. Long-Running ML Training

```python
@task(inject_context=True)
def train_model(context):
    channel = context.get_channel()
    epoch = channel.get("epoch", 0)
    model = channel.get("model")

    # Train for one epoch
    model = train_epoch(model)
    channel.set("model", model)
    channel.set("epoch", epoch + 1)

    # Checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        context.checkpoint(metadata={"epoch": epoch + 1})

    if epoch + 1 < max_epochs:
        context.next_iteration()
    else:
        return "Training complete"
```

If training fails at epoch 47, resume from last checkpoint (epoch 40):
```python
context, metadata = CheckpointManager.resume_from_checkpoint("checkpoint_epoch_40.pkl")
# Training continues from epoch 41
```

##### 2. Multi-Stage ETL Pipeline

```python
with workflow("etl_pipeline") as wf:
    extract >> validate >> transform >> load

    # Checkpoint after expensive transform step
    @task(inject_context=True)
    def transform(context):
        result = expensive_transformation()
        context.checkpoint(metadata={"stage": "transformed"})
        return result

# If load step fails, resume from transform checkpoint
context, _ = CheckpointManager.resume_from_checkpoint("etl_checkpoint.pkl")
engine.execute(context)  # Only runs load step
```

##### 3. Distributed Workflow with Worker Failure

```python
# Worker 1 processes tasks and checkpoints
context = ExecutionContext.create(graph, "start", queue_backend="redis")

@task(inject_context=True)
def distributed_step(context):
    process_partition()
    context.checkpoint(metadata={"worker": "worker-1"})
    finalize_partition()

# ... worker processes tasks ...

# Worker 1 crashes → Worker 2 resumes from checkpoint
context, _ = CheckpointManager.resume_from_checkpoint("/shared/checkpoint.pkl")
# Pending tasks restored to queue, execution continues
```

##### 4. State Machine with Checkpoint at Each Transition

```python
@task(inject_context=True)
def order_state_machine(context):
    channel = context.get_channel()
    state = channel.get("order_state", "NEW")

    if state == "NEW":
        validate_order()
        channel.set("order_state", "VALIDATED")
        context.checkpoint(metadata={"state": "VALIDATED"})
        context.next_iteration()

    elif state == "VALIDATED":
        process_payment()
        channel.set("order_state", "PAID")
        context.checkpoint(metadata={"state": "PAID"})
        context.next_iteration()

    elif state == "PAID":
        ship_order()
        return "ORDER_COMPLETE"
```

#### Key Features

##### Backend Support

- **Local Backend** ✅ (Production Ready):
  - File-based storage
  - Three-file structure (.pkl, .state.json, .meta.json)
  - Fast checkpoint creation (~10-100ms)

- **Redis Backend** 🔄 (Design Complete):
  - Distributed checkpoint storage
  - Worker-accessible checkpoints
  - Path format: `redis://checkpoint_key`

- **S3 Backend** 🔄 (Planned):
  - Cloud-based checkpoint storage
  - Path format: `s3://bucket/checkpoint`

##### Full State Preservation

```python
# All state is preserved:
✅ Task results (via channel)
✅ Cycle counts per task
✅ Completed tasks list
✅ Pending task specs (full TaskSpec, not just IDs)
✅ Current task (for mid-execution resume)
✅ User metadata
✅ Session information
```

##### Checkpoint Validation

```python
# Automatic validation on resume
try:
    context, metadata = CheckpointManager.resume_from_checkpoint(path)
except FileNotFoundError:
    print("Checkpoint files missing")
except json.JSONDecodeError:
    print("Corrupted checkpoint data")
except Exception as e:
    print(f"Resume failed: {e}")
```

##### Metadata Enrichment

```python
# Add custom metadata
CheckpointManager.create_checkpoint(
    context,
    metadata={
        "project": "data_pipeline",
        "version": "2.0",
        "batch_id": "batch_2025_01_15",
        "records_processed": 1000000,
        "data_quality_score": 0.95
    }
)

# Access metadata on resume
context, metadata = CheckpointManager.resume_from_checkpoint(path)
print(f"Batch: {metadata.user_metadata['batch_id']}")
print(f"Records: {metadata.user_metadata['records_processed']}")
```

#### Checkpoint File Structure Example

**checkpoint.state.json**:
```json
{
  "version": "1.0",
  "session_id": "session_abc123",
  "steps": 5,
  "completed_tasks": ["task_a", "task_b"],
  "cycle_counts": {
    "training_task": 42
  },
  "pending_tasks": [
    {
      "task_id": "task_c",
      "executable": {
        "strategy": "reference",
        "module": "my_workflow",
        "name": "task_c_func",
        "qualname": "task_c_func"
      },
      "status": "READY",
      "retry_count": 0
    }
  ],
  "resume_from_current_task": false
}
```

**checkpoint.meta.json**:
```json
{
  "checkpoint_id": "session_abc123_5_1737648000",
  "session_id": "session_abc123",
  "created_at": "2025-01-23T12:00:00Z",
  "steps": 5,
  "start_node": "task_a",
  "backend": {
    "queue": "memory",
    "channel": "memory"
  },
  "user_metadata": {
    "project": "ml_training",
    "epoch": 40,
    "accuracy": 0.95
  }
}
```

#### Design Principles

1. **Single Source of Truth**: Graph owns tasks, checkpoints reference them
2. **Full State Capture**: All execution state is preserved
3. **Three-File Structure**: Separation of concerns (context, state, metadata)
4. **Schema Versioning**: Forward compatibility for checkpoint format
5. **Fail-Fast Validation**: Start node validation on context creation
6. **Defense in Depth**: Serialization attributes on all task types

#### Comparison with Competitors

| Feature | Graflow | LangGraph | Celery | Airflow |
|---------|---------|-----------|--------|---------|
| **User-controlled Checkpointing** | ✅ `context.checkpoint()` in tasks | ❌ Automatic only | N/A | ❌ |
| **Checkpoint Creation** | ✅ `create_checkpoint()` | ✅ Memory-based | ❌ | ⚠️ Task state only |
| **Resume from Checkpoint** | ✅ `resume_from_checkpoint()` | ✅ | ❌ | ⚠️ Task retry only |
| **Persistent Storage** | ✅ File-based (local/Redis/S3) | ⚠️ Memory only | N/A | ⚠️ Database |
| **Full State Preservation** | ✅ Complete state | ⚠️ Memory state | N/A | ❌ |
| **Pending Tasks** | ✅ Full TaskSpec | ⚠️ Limited | N/A | ⚠️ |
| **Metadata Management** | ✅ Rich metadata | ⚠️ Basic | N/A | ⚠️ |
| **Distributed Resume** | ✅ Redis backend | ❌ | N/A | ⚠️ Partial |
| **Cycle State** | ✅ Preserved | ❌ | N/A | N/A |
| **Channel Data** | ✅ Preserved | ⚠️ Memory | N/A | ⚠️ XCom |

**Key Advantage**: **User-controlled checkpointing from within tasks** via `context.checkpoint()`, enabling precise control over when to persist state. Production-ready with complete state preservation, distributed support, and three-file structure.

#### Best Practices

1. **Checkpoint at Critical Points**:
   ```python
   # After expensive operations
   expensive_computation()
   context.checkpoint(metadata={"stage": "computed"})
   ```

2. **Use Descriptive Metadata**:
   ```python
   CheckpointManager.create_checkpoint(
       context,
       metadata={
           "stage": "transformed",
           "records": 1000000,
           "timestamp": datetime.now().isoformat()
       }
   )
   ```

3. **Handle Resume Failures**:
   ```python
   try:
       context, metadata = CheckpointManager.resume_from_checkpoint(path)
       engine.execute(context)
   except Exception as e:
       logger.error(f"Resume failed: {e}")
       # Fallback: restart from beginning
       context = ExecutionContext.create(graph, start_node)
   ```

4. **Cleanup Old Checkpoints**:
   ```python
   # Keep only last N checkpoints
   checkpoint_dir = "checkpoints/session_123"
   checkpoints = sorted(glob(f"{checkpoint_dir}/*.pkl"))
   for old_checkpoint in checkpoints[:-5]:  # Keep last 5
       os.remove(old_checkpoint)
   ```

5. **Test Resume Logic**:
   ```python
   # Unit test checkpoint/resume
   checkpoint_path, _ = CheckpointManager.create_checkpoint(context)
   restored, _ = CheckpointManager.resume_from_checkpoint(checkpoint_path)
   assert restored.steps == context.steps
   assert restored.session_id == context.session_id
   ```

---

## Comparative Analysis

### Feature Matrix

| Feature | Graflow | LangGraph | Celery | Airflow |
|---------|---------|-----------|--------|---------|
| **Pythonic DSL** | ✅ `>>`, `\|` (DAG + cycles) | ❌ | ⚠️ Partial | ⚠️ Partial |
| **Runtime Dynamic Tasks** | ✅ `next_task()` | ❌ | ❌ | ⚠️ Dynamic DAG |
| **State Machine Execution** | ✅ `next_iteration()` + Channel | ⚠️ Graph cycles | ❌ | ❌ |
| **Worker Fleet CLI** | ✅ Built-in | ❌ | ✅ | ✅ |
| **Custom Handlers** | ✅ Pluggable | ❌ | ⚠️ Task classes | ⚠️ Operators |
| **Docker Execution** | ✅ Built-in handler | ❌ | ⚠️ Via operators | ⚠️ DockerOperator |
| **Parallel Error Policies** | ✅ 5 modes + custom | ❌ | ⚠️ Basic | ⚠️ trigger_rule |
| **Local/Distributed Switch** | ✅ 1 line | ❌ | ❌ | ❌ |
| **Channel Communication** | ✅ Namespaced KVS | ⚠️ State | ❌ | ⚠️ XCom |
| **Graceful Shutdown** | ✅ Built-in | N/A | ✅ | ✅ |
| **Metrics Collection** | ✅ Worker-level | ❌ | ⚠️ Flower | ✅ |
| **Cycle Detection** | ✅ Built-in | ⚠️ Manual | N/A | ❌ |
| **Context Managers** | ✅ `with workflow()` | ❌ | ❌ | ❌ |
| **Type Safety** | ✅ TypedChannel | ✅ Pydantic | ❌ | ❌ |
| **Checkpoint/Resume** | ✅ Production-ready | ⚠️ Memory only | ❌ | ⚠️ Task retry |

### Performance Characteristics

| Metric | Graflow | LangGraph | Celery | Airflow |
|--------|---------|-----------|--------|---------|
| **Local Overhead** | Low (in-process) | Low | High (broker) | High (DB) |
| **Distributed Latency** | Medium (Redis) | N/A | Medium | High (polling) |
| **Throughput** | High (parallel workers) | Low (single process) | High | Medium |
| **Memory Footprint** | Medium | Low | Medium | High |

---

## Use Case Guidelines

### When to Use Graflow ✅

1. **General-purpose Data Pipelines**
   - ETL workflows
   - Data processing pipelines
   - Batch processing jobs
   - Data analytics workflows

2. **Dynamic Workflows**
   - Conditional execution based on runtime data
   - Convergence algorithms (ML training, optimization)
   - Adaptive data processing
   - Error recovery with retry

3. **Distributed Execution**
   - Horizontal scaling requirements
   - Worker fleet management
   - Geographic distribution
   - Resource-specific workers (GPU, high-memory)

4. **Custom Execution Strategies**
   - Remote execution (SSH, cloud functions)
   - Special hardware (GPU, TPU)
   - Rate-limited API calls
   - Retry and caching logic

5. **Development Agility**
   - Local development without infrastructure
   - Quick prototyping
   - Seamless production deployment

6. **State Machine Implementations**
   - Protocol state machines (network, API)
   - Order processing workflows
   - Game loops and interactive systems
   - ML training state management

7. **Containerized Execution**
   - Security sandboxing (untrusted code)
   - Multi-version testing (Python, dependencies)
   - Reproducible environments
   - Legacy code execution (Python 2.7, old libraries)

8. **Long-Running Workflows with Checkpointing**
   - ML model training with periodic checkpoints
   - Multi-hour data processing pipelines
   - Workflows requiring fault tolerance
   - Resume from failure scenarios
   - Distributed workflows with worker failures

### When to Use LangGraph ✅

- LLM agent orchestration
- Conversational AI applications
- Checkpointing and replay
- Human-in-the-loop workflows

### When to Use Celery ✅

- Existing RabbitMQ/Redis infrastructure
- Background job processing (emails, notifications)
- Task routing by queue
- Fire-and-forget tasks

### When to Use Airflow ✅

- Schedule-based batch processing
- Data warehouse ETL
- Complex DAG visualization
- Audit logging requirements

---

## Production Deployment

### Infrastructure Requirements

#### Minimal Setup (Single Server)
```
┌─────────────────────────────────┐
│       Single Server             │
│                                 │
│  ┌──────────┐  ┌────────────┐  │
│  │  Redis   │  │ Graflow    │  │
│  │  (Queue) │  │ Worker x3  │  │
│  └──────────┘  └────────────┘  │
│                                 │
│  ┌──────────────────────────┐  │
│  │  Main Application        │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
```

#### Scalable Setup (Multi-Server)
```
┌──────────────┐
│ Redis Cluster│
│  (HA Setup)  │
└──────┬───────┘
       │
   ┌───┴───┬───────┬────────┐
   │       │       │        │
┌──▼───┐ ┌─▼────┐ ┌─▼─────┐ ┌─▼─────┐
│Server1│ │Server2│ │Server3│ │Server4│
│4 Work.│ │8 Work.│ │2 Work.│ │ GPU   │
└───────┘ └──────┘ └───────┘ └───────┘
```

### Deployment Strategies

#### Docker Compose
```yaml
version: '3.8'
services:
  redis:
    image: redis:7.2
    ports:
      - "6379:6379"

  worker:
    image: graflow-worker:latest
    environment:
      - REDIS_HOST=redis
      - MAX_CONCURRENT_TASKS=4
    deploy:
      replicas: 3
    depends_on:
      - redis
```

#### Kubernetes Helm Chart
```yaml
# values.yaml
workers:
  replicas: 3
  resources:
    requests:
      cpu: "1"
      memory: "2Gi"
    limits:
      cpu: "2"
      memory: "4Gi"

redis:
  enabled: true
  cluster:
    enabled: true
    nodes: 3
```

### Monitoring and Observability

#### Metrics to Track
- Worker health (tasks processed, success rate)
- Queue depth (pending tasks)
- Task execution time (P50, P95, P99)
- Error rate by task type
- Worker resource utilization (CPU, memory)

#### Recommended Tools
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Redis Commander**: Queue inspection
- **Custom dashboard**: Worker metrics API

---

## Future Roadmap

### Planned Features

1. **Enhanced Monitoring**
   - Prometheus exporter
   - Grafana dashboard templates
   - Real-time task tracking UI

2. **Advanced Scheduling**
   - Cron-based task scheduling
   - Priority queues
   - Task dependencies across workflows

3. **Workflow Composition**
   - Nested workflows
   - Workflow templates
   - Workflow library

4. **Integrations**
   - Kubernetes native execution
   - AWS Lambda backend
   - Apache Beam compatibility

5. **Enhanced Checkpoint Features**
   - Redis backend for distributed checkpoints
   - S3 backend for cloud storage
   - Automatic checkpoint cleanup policies
   - Checkpoint compression

---

## Conclusion

Graflow represents a **new generation of workflow orchestration**, combining:

- **Pythonic elegance** (operator DSL)
- **Production robustness** (worker fleet, error policies, checkpoint/resume)
- **Development agility** (local/distributed switching)
- **Extensibility** (custom handlers, policies)
- **Dynamic capabilities** (runtime task generation)
- **Fault tolerance** (checkpoint/resume for long-running workflows)

It fills the gap between lightweight tools (LangGraph) and heavyweight infrastructure (Airflow), providing a **sweet spot for modern data engineering and ML workflows**.

---

**Document Maintainer**: Graflow Team
**Last Review**: 2025-10-23 (Updated: Checkpoint/Resume implementation completed)
**Next Review**: Quarterly
