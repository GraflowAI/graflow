# Dynamic Task Generation

This directory contains examples demonstrating dynamic task generation - creating tasks at both compile-time and runtime based on data and conditions.

## Overview

Dynamic task generation enables:
- **Flexible Workflows** - Adapt workflow structure based on configuration
- **Data-Driven Execution** - Create tasks based on runtime data
- **Conditional Logic** - Branch workflows dynamically
- **Iterative Processing** - Implement loops and convergence patterns
- **Scalable Patterns** - Handle variable-sized datasets

## Examples

### 1. dynamic_tasks.py
**Difficulty**: Advanced
**Time**: 20 minutes

Demonstrates compile-time dynamic task generation using loops and factories.

**Key Concepts**:
- Creating tasks in loops (compile-time)
- Task factory patterns
- Configuration-driven workflows
- Batch processing with dynamic tasks

**What You'll Learn**:
- How to generate tasks programmatically
- Building workflows from configuration
- Factory functions for tasks
- Dynamic DAG construction

**Run**:
```bash
python examples/07_dynamic_tasks/dynamic_tasks.py
```

**Expected Output**:
```
=== Dynamic Task Generation ===

Scenario 1: Batch Processing
Creating 5 processing tasks dynamically...
✅ Task process_item_0 created
✅ Task process_item_1 created
...

Scenario 2: Conditional Pipeline
Configuration: enable_validation=True
✅ Extract task added
✅ Validation task added (conditional)
```

**Real-World Applications**:
- Batch data processing
- Multi-tenant workflows
- A/B testing pipelines
- Feature-flag driven workflows

---

### 2. runtime_dynamic_tasks.py
**Difficulty**: Advanced
**Time**: 30 minutes

Comprehensive guide to runtime dynamic task generation using `next_task()` and `next_iteration()`.

**Key Concepts**:
- Runtime task creation with `context.next_task()`
- Iterative processing with `context.next_iteration()`
- Conditional task branching at runtime
- TaskWrapper for dynamic tasks
- Data-driven workflow adaptation
- State machines and convergence patterns

**What You'll Learn**:
- Creating tasks during execution based on runtime conditions
- Implementing iterative/convergence patterns
- Building state machines with `next_iteration()`
- Error recovery with retry logic
- When to use runtime vs compile-time generation

**Run**:
```bash
python examples/07_dynamic_tasks/runtime_dynamic_tasks.py
```

**Expected Output**:
```
=== Runtime Dynamic Task Generation ===

=== Scenario 1: Conditional Task Creation ===
Processing data: value=150
✅ Value > 100, creating high-value task
High-value processing: 150 → 300

=== Scenario 2: Iterative Processing ===
Iteration 0: accuracy=0.50
Iteration 1: accuracy=0.65
✅ Converged! Saving final model

=== Scenario 5: State Machine ===
Current state: START (data=0)
→ Transitioning to PROCESSING
...
```

**Real-World Applications**:
- ML model training with convergence
- Retry logic and error recovery
- State machine workflows
- Adaptive data processing

---

## Learning Path

**Recommended Order**:
1. Start with `dynamic_tasks.py` for compile-time generation
2. Move to `runtime_dynamic_tasks.py` for runtime patterns

**Prerequisites**:
- Complete examples from 01-03
- Understanding of workflow patterns from 02
- Familiarity with channels from 03

**Total Time**: ~50 minutes

---

## Key Concepts

### Compile-Time vs Runtime Generation

**Compile-Time** (`dynamic_tasks.py`):
- Tasks created during workflow definition
- Structure known before execution
- Uses Python loops and conditionals
- Good for: batch processing, configuration-driven workflows

```python
with workflow("batch") as ctx:
    # Create tasks in a loop (compile-time)
    for i in range(10):
        @task(id=f"process_{i}")
        def process():
            return process_item(i)
```

**Runtime** (`runtime_dynamic_tasks.py`):
- Tasks created during workflow execution
- Structure determined by data/results
- Uses `next_task()` and `next_iteration()`
- Good for: conditional branching, iterative processing

```python
@task(inject_context=True)
def classifier(context: TaskExecutionContext):
    if value > threshold:
        high_task = TaskWrapper("high", lambda: process_high())
        context.next_task(high_task)
    else:
        low_task = TaskWrapper("low", lambda: process_low())
        context.next_task(low_task)
```

---

## Common Patterns

### 1. Batch Processing (Compile-Time)

Process variable-sized datasets:

```python
with workflow("batch") as ctx:
    num_batches = len(data) // batch_size

    for i in range(num_batches):
        @task(id=f"batch_{i}")
        def process_batch():
            start = i * batch_size
            end = start + batch_size
            return process(data[start:end])
```

### 2. Conditional Branching (Runtime)

Branch based on runtime conditions:

```python
@task(inject_context=True)
def router(context: TaskExecutionContext):
    result = analyze_data()

    if result["type"] == "A":
        task_a = TaskWrapper("process_a", lambda: handle_type_a())
        context.next_task(task_a)
    else:
        task_b = TaskWrapper("process_b", lambda: handle_type_b())
        context.next_task(task_b)
```

### 3. Iterative Convergence (Runtime)

Implement convergence loops:

```python
@task(inject_context=True)
def optimize(context: TaskExecutionContext, params=None):
    if params is None:
        params = initialize_params()

    improved_params = optimization_step(params)

    if converged(improved_params):
        save_task = TaskWrapper("save", lambda: save_model(improved_params))
        context.next_task(save_task)
    else:
        context.next_iteration(improved_params)
```

### 4. State Machine (Runtime)

Build state-driven workflows:

```python
@task(inject_context=True)
def state_machine(context: TaskExecutionContext):
    state = context.get_channel().get("state", "START")

    if state == "START":
        context.get_channel().set("state", "PROCESSING")
        context.next_iteration()
    elif state == "PROCESSING":
        if processing_complete():
            context.get_channel().set("state", "END")
            end_task = TaskWrapper("end", lambda: finalize())
            context.next_task(end_task)
        else:
            context.next_iteration()
```

---

## Best Practices

### ✅ DO:

- Use compile-time generation for known structures
- Use runtime generation for data-dependent logic
- Set unique IDs for dynamically created tasks
- Use `max_steps` to prevent infinite loops
- Document dynamic behavior clearly
- Test with various data scenarios

### ❌ DON'T:

- Create tasks with duplicate IDs
- Use runtime generation when compile-time works
- Forget to set termination conditions
- Create deeply nested dynamic tasks
- Capture large objects in task closures

---

## Troubleshooting

### Infinite Loops

**Problem**: Workflow never completes

**Solution**:
```python
# Always set max_steps
ctx.execute("start", max_steps=100)

# Add termination condition
iteration_count = context.get_channel().get("iteration", 0)
if iteration_count > 10:
    # Force termination
    return
```

### Duplicate Task IDs

**Problem**: `DuplicateTaskError`

**Solution**:
```python
# Use unique IDs with counters or UUIDs
import uuid

task_id = f"dynamic_task_{uuid.uuid4().hex[:8]}"
task = TaskWrapper(task_id, lambda: process())
```

### Memory Issues

**Problem**: Too many dynamic tasks created

**Solution**:
```python
# Limit number of dynamic tasks
max_tasks = 100
if task_count < max_tasks:
    create_dynamic_task()
else:
    handle_overflow()
```

---

## Real-World Use Cases

### 1. ML Hyperparameter Tuning

Iteratively optimize parameters:

```python
@task(inject_context=True)
def tune_hyperparameters(context, params=None):
    if params is None:
        params = initial_grid_search()

    score = evaluate_model(params)

    if score > target_score or iterations > max_iterations:
        final_task = TaskWrapper("deploy", lambda: deploy_model(params))
        context.next_task(final_task)
    else:
        improved = bayesian_optimization(params, score)
        context.next_iteration(improved)
```

### 2. Multi-Tenant Data Processing

Create tasks per tenant:

```python
with workflow("multi_tenant") as ctx:
    tenants = get_active_tenants()

    for tenant_id in tenants:
        @task(id=f"process_{tenant_id}")
        def process_tenant():
            data = fetch_tenant_data(tenant_id)
            return process_data(data)
```

### 3. Adaptive ETL Pipeline

Adjust pipeline based on data:

```python
@task(inject_context=True)
def adaptive_transform(context):
    data = context.get_channel().get("data")

    if data["quality"] < 0.8:
        # Add extra cleaning step
        clean_task = TaskWrapper("deep_clean", lambda: deep_clean(data))
        context.next_task(clean_task)
    else:
        # Skip to validation
        val_task = TaskWrapper("validate", lambda: validate(data))
        context.next_task(val_task)
```

---

## Performance Considerations

### Task Creation Overhead

- Compile-time: ~1-5ms per task
- Runtime: ~5-10ms per task (includes graph update)

### Memory Usage

- Each task: ~1-5KB
- 1000 dynamic tasks: ~1-5MB
- Use batch processing for large datasets

### Optimization Tips

1. **Reuse task functions**: Define once, call with different parameters
2. **Batch operations**: Group similar tasks
3. **Limit depth**: Avoid deeply nested dynamic creation
4. **Clean up**: Remove unnecessary data from channels

---

## Integration with Other Features

### With Distributed Execution

Dynamic tasks work with Redis distribution:

```python
ctx.execute(
    "start",
    queue_backend=QueueBackend.REDIS,
    max_steps=1000
)
```

### With Channels

Use channels for state management:

```python
@task(inject_context=True)
def dynamic_processor(context):
    channel = context.get_channel()
    state = channel.get("state", {})

    # Update state
    state["processed"] += 1
    channel.set("state", state)

    # Decide next step based on state
    if state["processed"] < total:
        context.next_iteration()
```

---

## Next Steps

After mastering dynamic task generation, explore:

1. **Workflow Composition** (`../08_workflow_composition/`)
   - Combine dynamic tasks with concurrent workflows
   - Build reusable workflow templates

2. **Real-World Examples** (`../09_real_world/`)
   - Apply dynamic patterns to production use cases
   - See complete implementations

3. **Custom Development**
   - Build your own dynamic workflow patterns
   - Create domain-specific task generators

---

## Additional Resources

- **TaskWrapper API**: See `graflow/core/task.py`
- **ExecutionContext**: See `graflow/core/context.py`
- **Advanced Patterns**: `../06_advanced/`

---

**Ready to build dynamic workflows?** Start with `dynamic_tasks.py`!
