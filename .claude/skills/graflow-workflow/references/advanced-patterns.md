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
from graflow.hitl.manager import FeedbackManager
from graflow.hitl.types import FeedbackType

@task(inject_context=True)
def approval_task(ctx: TaskExecutionContext):
    manager = FeedbackManager()

    feedback = manager.request_feedback(
        request_id="deploy_approval",
        feedback_type=FeedbackType.APPROVAL,
        message="Approve deployment to production?",
        timeout=300  # 5 minutes
    )

    if feedback.approved:
        ctx.next_task(deploy())
    else:
        ctx.cancel_workflow(error="Deployment rejected")
```

### Text Input
```python
feedback = manager.request_feedback(
    request_id="config_input",
    feedback_type=FeedbackType.TEXT,
    message="Enter configuration value:",
    timeout=60
)
config_value = feedback.value
```

### Selection
```python
feedback = manager.request_feedback(
    request_id="model_selection",
    feedback_type=FeedbackType.SELECTION,
    message="Select model to use:",
    options=["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"],
    timeout=120
)
selected_model = feedback.value
```

### Channel Integration
```python
@task(inject_context=True)
def feedback_to_channel(ctx: TaskExecutionContext):
    manager = FeedbackManager()

    feedback = manager.request_feedback(
        request_id="user_input",
        feedback_type=FeedbackType.TEXT,
        message="Enter search query:",
        channel_key="search_query",  # Auto-write to channel
        execution_context=ctx.execution_context
    )
    # feedback.value is also written to channel["search_query"]
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
