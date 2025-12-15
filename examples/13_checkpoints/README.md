# Checkpoint/Resume Examples

This directory contains examples demonstrating Graflow's checkpoint/resume functionality for workflow state persistence and recovery.

## Overview

Checkpoints allow you to:
- Save workflow state at specific points during execution
- Resume execution from a saved checkpoint after interruption or failure
- Implement fault tolerance for long-running workflows
- Support state machine workflows with iterative checkpoints

## Key Concepts

### Checkpoint Structure
Each checkpoint consists of three files:
- `.pkl` - Serialized ExecutionContext (graph, channel data)
- `.state.json` - Execution state (steps, completed tasks, pending tasks, cycle counts)
- `.meta.json` - Metadata (checkpoint ID, timestamps, user metadata)

### Checkpoint Creation
Checkpoints are created in two steps:
1. **Request**: Task calls `context.checkpoint(metadata={...})` to set a flag
2. **Creation**: Engine creates checkpoint AFTER task completes successfully

This deferred execution ensures tasks are marked as completed in the checkpoint.

### Resuming
Use `CheckpointManager.resume_from_checkpoint(path)` to restore workflow state and continue execution.

## Examples

### 1. Basic Checkpoint/Resume (`01_basic_checkpoint.py`)
**Simplest example** showing fundamental checkpoint/resume workflow.
- Create checkpoint after task completion
- Resume from checkpoint
- Continue execution from saved state

**When to use**: Learning checkpoint basics, simple workflows

```bash
uv run python examples/13_checkpoints/01_basic_checkpoint.py
```

### 2. State Machine with Checkpoints (`02_state_machine_checkpoint.py`)
**Production pattern** for state-based workflows with checkpoint at each state transition.
- Order processing: NEW → VALIDATED → PAID → SHIPPED
- Checkpoint after each state transition
- Resume from any state

**When to use**: Order processing, approval workflows, multi-stage pipelines

```bash
uv run python examples/13_checkpoints/02_state_machine_checkpoint.py
```

### 3. Periodic Checkpoints (`03_periodic_checkpoint.py`)
**Long-running workflows** with periodic checkpoints.
- ML training simulation with iterative checkpoints
- Checkpoint every N iterations
- Resume from latest checkpoint

**When to use**: ML training, batch processing, data pipelines

```bash
uv run python examples/13_checkpoints/03_periodic_checkpoint.py
```

### 4. Fault Recovery (`04_fault_recovery.py`)
**Fault tolerance** demonstration with simulated failures.
- Execute workflow with potential failure points
- Create checkpoint before expensive operations
- Resume after simulated failure

**When to use**: Unreliable infrastructure, expensive operations, production workflows

```bash
uv run python examples/13_checkpoints/04_fault_recovery.py
```

## Idempotency: Critical User Responsibility

### How Task Execution Works with Checkpoint/Resume

**Important**: When resuming from a checkpoint, **tasks always re-execute from the beginning**.

This is by design for the following reasons:
- Saving intermediate task state is complex and error-prone
- Fully restoring local variables and stack state of task functions is difficult
- Keeping only consistent states (before or after task completion) keeps the system simple

### User Responsibility: Design Idempotent Tasks

**Designing idempotent tasks is the user's responsibility.**

Idempotency means: Executing the same task multiple times produces the same result as executing it once.

#### Problems with Non-Idempotent Tasks

```python
# ❌ Bad Example: Non-idempotent
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task

@task(inject_context=True)
def process_orders(task_ctx: TaskExecutionContext):
    # Resuming from checkpoint will process the same orders twice
    orders = fetch_new_orders()
    for order in orders:
        charge_customer(order)  # Double charge on re-execution!
        ship_product(order)     # Double shipment on re-execution!
    task_ctx.checkpoint()
```

If this task crashes and resumes from checkpoint:
- Customers are charged twice for the same order
- Products are shipped twice
- Data inconsistencies occur

#### How to Achieve Idempotency

### 1. Channel-Based State Management (Recommended)

```python
# ✅ Good Example: Track state in channel
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task

@task(inject_context=True)
def process_orders(task_ctx: TaskExecutionContext):
    channel = task_ctx.get_channel()
    processed_order_ids = channel.get("processed_order_ids", set())

    orders = fetch_new_orders()
    for order in orders:
        # Skip if already processed
        if order.id in processed_order_ids:
            continue

        charge_customer(order)
        ship_product(order)

        # Mark as processed
        processed_order_ids.add(order.id)
        channel.set("processed_order_ids", processed_order_ids)

    task_ctx.checkpoint()
```

**Key Points**:
- Store processed information in channel
- On re-execution, read state from channel and skip already-completed work
- Channel data is included in checkpoint, so it persists across resume

### 2. State Machine Pattern

```python
# ✅ Good Example: State-based execution control
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task

@task(inject_context=True)
def multi_stage_task(task_ctx: TaskExecutionContext):
    channel = task_ctx.get_channel()
    state = channel.get("state", "INIT")

    if state == "INIT":
        initialize_resources()
        channel.set("state", "PROCESSING")
        task_ctx.checkpoint()
        task_ctx.next_iteration()
    elif state == "PROCESSING":
        process_data()
        channel.set("state", "FINALIZING")
        task_ctx.checkpoint()
        task_ctx.next_iteration()
    elif state == "FINALIZING":
        finalize()
        return "COMPLETE"
```

**Key Points**:
- Check state at the beginning of each stage
- Skip already-completed stages
- Resume from incomplete stage

### 3. Use Idempotency Features of External Systems

```python
# ✅ Good Example: Idempotent API calls
import uuid
from graflow.core.context import TaskExecutionContext
from graflow.core.decorators import task

@task(inject_context=True)
def call_external_api(task_ctx: TaskExecutionContext):
    channel = task_ctx.get_channel()

    # Use idempotency key (same key ensures idempotent re-execution)
    idempotency_key = channel.get("idempotency_key")
    if not idempotency_key:
        idempotency_key = str(uuid.uuid4())
        channel.set("idempotency_key", idempotency_key)

    # Many payment APIs support idempotency keys
    result = payment_api.charge(
        amount=100,
        idempotency_key=idempotency_key  # Same key prevents duplicate execution
    )

    task_ctx.checkpoint()
    return result
```

### Idempotency Checklist

When implementing tasks, verify the following:

- [ ] Is the task safe to re-execute?
- [ ] Will writes to external resources (DB, API, files) not duplicate?
- [ ] Are you tracking processed state in the channel?
- [ ] Are you using idempotency features of external systems (idempotency keys, etc.)?
- [ ] For financial operations (billing, payments), do you prevent double execution?

### Summary

**Graflow's Responsibility**:
- Creating and restoring checkpoints
- Saving task graph and channel data
- Accurate resumption from checkpoints

**User's Responsibility**:
- **Designing tasks to be idempotent**
- Implementing state management using channels
- Safe integration with external systems

Proper use of checkpoint functionality requires idempotent task design.

## Best Practices

### 1. Checkpoint After Expensive Operations
```python
@task(inject_context=True)
def expensive_task(task_ctx):
    expensive_computation()
    task_ctx.checkpoint(metadata={"stage": "computation_complete"})
    # If workflow crashes here, computation is saved
```

### 2. Include Meaningful Metadata
```python
task_ctx.checkpoint(metadata={
    "stage": "validation_complete",
    "records_processed": 10000,
    "timestamp": time.time()
})
```

### 3. Auto-Generated vs Custom Paths
```python
# Auto-generated (recommended for most cases)
task_ctx.checkpoint()  # Path: checkpoints/{session_id}/...

# Custom path (for specific requirements)
task_ctx.checkpoint(path="/tmp/my_checkpoint")
```

## Architecture Notes

### Deferred Checkpoint Execution
When `context.checkpoint()` is called during task execution:
1. Sets `context.checkpoint_requested = True`
2. Stores user metadata in `context.checkpoint_request_metadata`
3. Task continues to completion
4. **Engine creates checkpoint AFTER task completes**
5. Task is marked as completed in checkpoint state

This ensures:
- Tasks are either completed or pending (no "currently executing" state)
- Checkpoints represent consistent states
- Resume continues from next pending task

### What Gets Saved

**ExecutionContext (.pkl)**:
- Task graph structure
- Channel data (MemoryChannel) or session ID (RedisChannel)
- Backend configuration

**Checkpoint State (.state.json)**:
- Completed task IDs
- Pending TaskSpecs (full task specifications, not just IDs)
- Cycle counts
- Steps executed
- Schema version

**Metadata (.meta.json)**:
- Checkpoint ID, session ID
- Created timestamp (ISO 8601)
- User-defined metadata

### Results Storage
Task results are stored in the **channel** with key `{task_id}.__result__`:
- MemoryChannel: Saved in `.pkl` file
- RedisChannel: Persisted in Redis automatically

## Common Patterns

### Pattern 1: Multi-Stage Pipeline
```python
stage1 >> checkpoint >> stage2 >> checkpoint >> stage3
```
Checkpoint after each major stage for recovery.

### Pattern 2: Approval Workflow
```python
process >> request_approval >> [checkpoint on timeout] >> deploy
```
Checkpoint when waiting for human approval.

### Pattern 3: Iterative Training
```python
for epoch in epochs:
    train()
    if epoch % 10 == 0:
        checkpoint()
```
Periodic checkpoints during long-running training.

## Troubleshooting

### Checkpoint Files Not Created
- Ensure task completes successfully (checkpoint created AFTER completion)
- Check write permissions for checkpoint directory
- Verify no exceptions during task execution

### Resume Fails
- Check all three checkpoint files exist (`.pkl`, `.state.json`, `.meta.json`)
- Verify files are not corrupted
- Ensure same Python environment and Graflow version

### Task Re-executes on Resume
- Expected behavior: tasks restart from beginning on resume
- Use channel-based state to skip already-completed work
- See state machine pattern examples

## Related Documentation

- Design: `docs/checkpoint/checkpoint_resume_design.md`
- Implementation: `docs/checkpoint/checkpoint_implementation_summary.md`
- Unit Tests: `tests/core/test_checkpoint.py`
- Scenario Tests: `tests/scenario/test_checkpoint_scenarios.py`
