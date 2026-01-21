# Advanced Graflow Patterns

## Table of Contents
- [Dynamic Task Generation](#dynamic-task-generation)
- [Fan-Out and Fan-In Patterns](#fan-out-and-fan-in-patterns)
- [Checkpoint and Resume](#checkpoint-and-resume)
- [Human-in-the-Loop (HITL)](#human-in-the-loop-hitl)
- [Prompt Management](#prompt-management)
- [Distributed Execution](#distributed-execution)

---

## Dynamic Task Generation

### next_task() - Add Task at Runtime
```python
from graflow.core.context import TaskExecutionContext

@task(inject_context=True)
def dispatcher(ctx: TaskExecutionContext):
    data = ctx.get_channel().get("data")

    if data["type"] == "priority":
        ctx.next_task(priority_handler(data=data))
    else:
        ctx.next_task(normal_handler(data=data))
```

### next_iteration() - Re-execute with New Parameters
```python
@task(inject_context=True)
def retry_task(ctx: TaskExecutionContext, attempt: int = 0):
    try:
        result = do_work()
        return result
    except TemporaryError:
        if attempt < 3:
            ctx.next_iteration(retry_task(attempt=attempt + 1))
        else:
            raise
```

### terminate_workflow() - Early Normal Exit
```python
@task(inject_context=True)
def check_complete(ctx: TaskExecutionContext):
    if all_done():
        ctx.terminate_workflow(result="completed early")
        return
    ctx.next_task(continue_processing())
```

### cancel_workflow() - Abort with Error
```python
@task(inject_context=True)
def validate(ctx: TaskExecutionContext):
    if critical_error():
        ctx.cancel_workflow(error="Critical validation failed")
```

---

## Fan-Out and Fan-In Patterns

### Dynamic Parallel Group
```python
from graflow.core.task import parallel

@task(inject_context=True)
def create_workers(ctx: TaskExecutionContext):
    items = ctx.get_channel().get("items")

    # Create parallel group dynamically
    workers = [
        process_item(task_id=f"worker_{i}", item=item)
        for i, item in enumerate(items)
    ]
    ctx.next_task(parallel(*workers))
```

### Fan-Out Pattern (Multiple Endpoints)
```python
@task(inject_context=True)
def branch(ctx: TaskExecutionContext, parent_path: str = ""):
    current_path = f"{parent_path}.branch" if parent_path else "branch"

    # Each branch independently triggers its endpoint
    ctx.next_task(parallel(
        process_a(parent_path=current_path),
        process_b(parent_path=current_path)
    ))

@task(inject_context=True)
def process_a(ctx: TaskExecutionContext, parent_path: str = ""):
    # ...
    ctx.next_task(endpoint(parent_path=f"{parent_path}.process_a"))

# Result: endpoint runs TWICE (once per branch)
```

### Fan-Out-then-Fan-In (Single Endpoint)
```python
@task(inject_context=True)
def root(ctx: TaskExecutionContext):
    # Use >> operator to chain parallel group with convergence point
    parallel_group = parallel(
        branch_a(task_id="branch_a"),
        branch_b(task_id="branch_b")
    )
    # integrator runs ONCE after both branches complete
    chained = parallel_group >> integrator(task_id="integrator")
    ctx.next_task(chained)
```

---

## Checkpoint and Resume

### Basic Checkpoint
```python
@task(inject_context=True)
def long_running(ctx: TaskExecutionContext):
    for i, batch in enumerate(batches):
        process(batch)

        # Create checkpoint after each batch
        if i % 10 == 0:
            ctx.checkpoint()  # Schedules checkpoint after task
```

### Resume from Checkpoint
```python
from graflow.core.checkpoint import CheckpointManager

# Resume workflow from checkpoint
manager = CheckpointManager(checkpoint_dir="./checkpoints")
result = manager.resume_from_checkpoint("checkpoint_path.pkl")
```

### Periodic Checkpoint
```python
@task(inject_context=True)
def periodic_checkpoint_task(ctx: TaskExecutionContext):
    import time
    last_checkpoint = time.time()

    for item in items:
        process(item)

        # Checkpoint every 5 minutes
        if time.time() - last_checkpoint > 300:
            ctx.checkpoint()
            last_checkpoint = time.time()
```

### Fault Recovery Pattern
```python
with workflow("fault_tolerant") as wf:
    try:
        result = wf.execute("entry_task", checkpoint_dir="./checkpoints")
    except Exception as e:
        print(f"Failed: {e}, resuming from checkpoint...")
        manager = CheckpointManager(checkpoint_dir="./checkpoints")
        result = manager.resume_from_latest()
```

---

## Human-in-the-Loop (HITL)

### Basic Approval
```python
@task(inject_context=True)
def approval_task(ctx: TaskExecutionContext):
    response = ctx.request_feedback(
        feedback_type="approval",
        prompt="Approve deployment to production?",
        timeout=300.0,  # 5 minutes
    )

    if response.approved:
        ctx.next_task(deploy())
    else:
        ctx.cancel_workflow(error=f"Deployment rejected: {response.reason}")
```

### Text Input
```python
@task(inject_context=True)
def text_input_task(ctx: TaskExecutionContext):
    response = ctx.request_feedback(
        feedback_type="text",
        prompt="Enter configuration value:",
        timeout=60.0,
    )
    config_value = response.text
```

### Selection
```python
@task(inject_context=True)
def selection_task(ctx: TaskExecutionContext):
    response = ctx.request_feedback(
        feedback_type="selection",
        prompt="Select model to use:",
        options=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
        timeout=120.0,
    )
    selected_model = response.selected
```

### Channel Integration
```python
@task(inject_context=True)
def feedback_to_channel(ctx: TaskExecutionContext):
    response = ctx.request_feedback(
        feedback_type="selection",
        prompt="Select processing mode:",
        options=["low_cost", "balanced", "high_performance"],
        channel_key="processing_mode",  # Auto-write to channel
        write_to_channel=True,
        timeout=60.0,
    )
    # response.selected is also written to channel["processing_mode"]
```

### Timeout and Checkpoint
```python
@task(inject_context=True)
def review_task(ctx: TaskExecutionContext):
    try:
        response = ctx.request_feedback(
            feedback_type="text",
            prompt="Review findings and provide your assessment:",
            timeout=2.0,  # Short timeout triggers checkpoint
        )
        return response.text
    except Exception as e:
        # Timeout triggers checkpoint creation
        # Workflow can resume when feedback is provided via API
        raise
```

---

## Prompt Management

### YAML Prompts
```python
from graflow.prompts.yaml_manager import YAMLPromptManager

manager = YAMLPromptManager("prompts/")

# Load and render prompt
prompt = manager.get_prompt("summarize", label="production")
rendered = prompt.render(text="content to summarize", max_words=100)
```

### YAML Prompt File Format
```yaml
# prompts/summarize.yaml
- label: production
  version: 1
  template: |
    Summarize the following text in {{max_words}} words or less:

    {{text}}

- label: development
  version: 2
  template: |
    [DEV] Summarize: {{text}}
```

### Prompts in Workflow
```python
@task(inject_context=True)
def summarize_task(ctx: TaskExecutionContext, text: str):
    manager = YAMLPromptManager("prompts/")
    prompt = manager.get_prompt("summarize", label="production")

    messages = [{"role": "user", "content": prompt.render(text=text)}]
    # Use with LLM client...
```

---

## Distributed Execution

### Redis Backend Setup
```python
from graflow.queue.factory import QueueBackend, TaskQueueFactory
from graflow.channels.factory import ChannelFactory

# Create distributed context
queue = TaskQueueFactory.create(
    backend=QueueBackend.REDIS,
    queue_name="my_workflow"
)
channel = ChannelFactory.create(
    backend="redis",
    channel_name="my_channel"
)
```

### Worker Process
```bash
# Start workers
python -m graflow.worker.main --worker-id worker-1
python -m graflow.worker.main --worker-id worker-2
```

### Distributed Workflow
```python
from graflow.core.context import create_execution_context

context = create_execution_context(
    queue_backend=QueueBackend.REDIS,
    channel_backend="redis",
    queue_name="distributed_pipeline"
)

with workflow("distributed", execution_context=context) as wf:
    fetch >> (process_a | process_b | process_c) >> aggregate
    wf.execute("fetch")
```

### Redis Coordinator for Parallel Groups
```python
from graflow.coordination.redis import RedisCoordinator

coordinator = RedisCoordinator(redis_url="redis://localhost:6379")

# Parallel group execution with distributed coordination
group = ParallelGroup(
    [task_a, task_b, task_c],
    name="distributed_group",
    coordinator=coordinator
)
```
