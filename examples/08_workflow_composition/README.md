# Workflow Composition

This directory contains examples demonstrating workflow composition patterns - creating reusable workflow templates and running multiple workflows concurrently.

## Overview

Workflow composition enables:
- **Concurrent Execution** - Run multiple workflow instances in parallel
- **Reusable Templates** - Create parameterized workflow factories
- **Scalable Patterns** - Handle multiple independent workflows
- **Factory Patterns** - Build workflows from configuration
- **Thread-Safe Execution** - Isolated workflow contexts

## Examples

### 1. concurrent_workflows.py
**Difficulty**: Advanced
**Time**: 25 minutes

Demonstrates concurrent execution of multiple workflow instances using Python's threading module.

**Key Concepts**:
- Running multiple workflow contexts concurrently
- Thread-safe workflow execution
- Managing multiple independent workflow instances
- Coordinating concurrent workflow completion
- Use cases for concurrent workflows

**What You'll Learn**:
- How to run workflows in parallel threads
- Thread safety and isolation
- Batch processing with concurrent workers
- When to use threading vs distributed execution
- Performance considerations

**Run**:
```bash
python examples/08_workflow_composition/concurrent_workflows.py
```

**Expected Output**:
```
=== Concurrent Workflow Execution ===

Starting 3 concurrent workers...

Worker 0: Starting work
Worker 1: Starting work
Worker 2: Starting work
...
✅ All 3 workers completed successfully
```

**Real-World Applications**:
- Multi-tenant processing
- Parallel data pipelines
- A/B testing multiple models
- Batch processing with parallel batches

---

### 2. workflow_factory.py
**Difficulty**: Advanced
**Time**: 25 minutes

Demonstrates the workflow factory pattern for creating reusable workflow templates.

**Key Concepts**:
- Creating workflow factory functions
- Parameterized workflow generation
- Workflow template reuse
- Multiple instances of the same workflow pattern
- Workflow composition patterns

**What You'll Learn**:
- How to create workflow factories
- Parameterizing workflows
- Building reusable templates
- Composing complex workflows from simple factories
- Testing with different configurations

**Run**:
```bash
python examples/08_workflow_composition/workflow_factory.py
```

**Expected Output**:
```
=== Workflow Factory Pattern ===

Creating ETL pipeline instances...

ETL Pipeline 1: Loading data
ETL Pipeline 2: Loading data
...
✅ Multiple instances executed independently
```

**Real-World Applications**:
- Multi-customer processing
- Environment-specific workflows
- A/B testing pipelines
- Standardized workflow templates

---

## Learning Path

**Recommended Order**:
1. Start with `workflow_factory.py` to understand template creation
2. Move to `concurrent_workflows.py` to learn parallel execution

**Prerequisites**:
- Complete examples from 01-03 (basics, workflows, data flow)
- Understanding of workflow patterns from 02_workflows
- Familiarity with Python threading (helpful for concurrent_workflows)

**Total Time**: ~50 minutes

---

## Key Concepts

### Workflow Factory Pattern

Create reusable workflow templates:

```python
def create_pipeline(name: str, source: str) -> WorkflowContext:
    """Factory for ETL pipelines."""
    ctx = workflow(name)

    with ctx:
        @task
        def load():
            print(f"Loading from {source}")

        @task
        def transform():
            print("Transforming data")

        @task
        def save():
            print("Saving results")

        load >> transform >> save

    return ctx

# Create multiple instances
pipeline1 = create_pipeline("pipeline_1", source="database")
pipeline2 = create_pipeline("pipeline_2", source="api")

# Execute independently
pipeline1.execute("load")
pipeline2.execute("load")
```

### Concurrent Workflow Execution

Run multiple workflow instances in parallel:

```python
import threading

def worker_workflow(worker_id: int):
    """Execute a workflow in a worker thread."""
    with workflow(f"worker_{worker_id}") as ctx:
        @task
        def process():
            print(f"Worker {worker_id}: Processing")

        ctx.execute("process")

# Create and start worker threads
threads = []
for i in range(5):
    thread = threading.Thread(target=worker_workflow, args=(i,))
    threads.append(thread)
    thread.start()

# Wait for completion
for thread in threads:
    thread.join()
```

---

## Common Patterns

### 1. Multi-Tenant Processing

Process workflows for multiple customers:

```python
def process_customer_data(customer_id: str):
    """Process data for a specific customer."""
    with workflow(f"customer_{customer_id}") as ctx:
        @task
        def load_customer_data():
            return load_data(customer_id)

        @task
        def process_data():
            return process(customer_id)

        load_customer_data >> process_data
        ctx.execute("load_customer_data")

# Process multiple customers concurrently
customers = ["cust_001", "cust_002", "cust_003"]
threads = [Thread(target=process_customer_data, args=(cid,))
           for cid in customers]

for t in threads:
    t.start()
for t in threads:
    t.join()
```

### 2. Environment-Specific Workflows

Create workflows based on environment:

```python
def create_pipeline(env: str):
    """Create pipeline for specific environment."""
    if env == "prod":
        return create_prod_pipeline()
    elif env == "staging":
        return create_staging_pipeline()
    else:
        return create_dev_pipeline()

prod_pipeline = create_pipeline("prod")
staging_pipeline = create_pipeline("staging")
```

### 3. A/B Testing

Run multiple workflow variants concurrently:

```python
def run_experiment(variant: str):
    """Run workflow for specific variant."""
    with workflow(f"variant_{variant}") as ctx:
        # Define variant-specific workflow
        pass

threads = [
    Thread(target=run_experiment, args=("control",)),
    Thread(target=run_experiment, args=("variant_a",)),
    Thread(target=run_experiment, args=("variant_b",))
]
```

### 4. Batch Processing with Parallel Batches

Process batches concurrently:

```python
def process_batch(batch_id: int, items: list):
    """Process a batch of items."""
    with workflow(f"batch_{batch_id}") as ctx:
        @task
        def process_items():
            for item in items:
                process(item)

        ctx.execute("process_items")

# Split data into batches
batches = partition_data(data, batch_size=100)

# Process batches concurrently
threads = [Thread(target=process_batch, args=(i, batch))
           for i, batch in enumerate(batches)]
```

---

## Best Practices

### ✅ DO:

- Use factories for reusable workflow patterns
- Keep workflows independent when running concurrently
- Use unique workflow names for each instance
- Parameterize workflows for flexibility
- Test factories with various configurations
- Use thread-safe data structures for shared results
- Consider multiprocessing for CPU-bound tasks
- Use Redis workers for distributed execution

### ❌ DON'T:

- Share mutable state between concurrent workflows
- Use the same workflow name for multiple instances
- Run too many threads (match CPU cores for CPU-bound)
- Forget error handling in concurrent workflows
- Mix factory pattern with hard-coded values
- Create workflows without cleanup/finalization

---

## Troubleshooting

### Thread Safety Issues

**Problem**: Race conditions or shared state corruption

**Solution**:
```python
# ✅ Good - isolated state
def worker(worker_id):
    with workflow(f"worker_{worker_id}") as ctx:
        # Each workflow has its own context
        pass

# ❌ Bad - shared mutable state
shared_list = []
def worker(worker_id):
    shared_list.append(value)  # Race condition!
```

### Performance Issues

**Problem**: Too many concurrent workflows slow down execution

**Solution**:
```python
from concurrent.futures import ThreadPoolExecutor

# Limit concurrent workers
max_workers = 5
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(workflow_func, i)
               for i in range(100)]
```

### Factory Complexity

**Problem**: Factory functions become too complex

**Solution**:
```python
# Split into smaller factories
def create_load_tasks(ctx, source):
    @task
    def load():
        # Load logic
        pass

def create_transform_tasks(ctx, method):
    @task
    def transform():
        # Transform logic
        pass

def create_pipeline(name, source, method):
    ctx = workflow(name)
    with ctx:
        create_load_tasks(ctx, source)
        create_transform_tasks(ctx, method)
    return ctx
```

---

## Real-World Use Cases

### 1. Multi-Customer Data Processing

```python
def create_customer_pipeline(customer_id: str):
    """Create pipeline for customer data processing."""
    ctx = workflow(f"customer_{customer_id}")

    with ctx:
        @task
        def extract_customer_data():
            return fetch_data(customer_id)

        @task
        def transform_data():
            return apply_transformations()

        @task
        def load_to_warehouse():
            return save_to_warehouse(customer_id)

        extract_customer_data >> transform_data >> load_to_warehouse

    return ctx

# Process all customers concurrently
customers = get_active_customers()
threads = [Thread(target=lambda c: create_customer_pipeline(c).execute("extract_customer_data"),
                  args=(cust,)) for cust in customers]
```

### 2. Parallel ML Model Training

```python
def create_model_training_workflow(model_id: str, config: dict):
    """Factory for model training workflows."""
    ctx = workflow(f"model_{model_id}")

    with ctx:
        @task
        def prepare_data():
            return load_training_data(config["dataset"])

        @task
        def train_model():
            return train(config["algorithm"], config["epochs"])

        @task
        def evaluate():
            return evaluate_model()

        prepare_data >> train_model >> evaluate

    return ctx

# Train multiple models concurrently
models = [
    {"id": "linear", "dataset": "data_v1", "algorithm": "linear", "epochs": 10},
    {"id": "neural", "dataset": "data_v1", "algorithm": "neural", "epochs": 50},
    {"id": "ensemble", "dataset": "data_v2", "algorithm": "ensemble", "epochs": 30}
]
```

---

## Performance Considerations

### Threading vs Multiprocessing vs Redis Workers

**Threading** (concurrent_workflows.py):
- ✅ Simple API, low overhead
- ✅ Good for I/O-bound workflows
- ❌ Python GIL limits CPU parallelism
- ❌ Limited to one machine

**Multiprocessing**:
- ✅ True parallelism for CPU-bound tasks
- ✅ Better CPU utilization
- ❌ Higher memory overhead
- ❌ More complex inter-process communication

**Redis Workers** (see 05_distributed):
- ✅ Scales across multiple machines
- ✅ Better fault tolerance
- ✅ Can handle thousands of tasks
- ❌ Requires Redis infrastructure
- ❌ More complex setup

### Optimization Tips

1. **Choose the right concurrency model**:
   - I/O-bound → Threading
   - CPU-bound → Multiprocessing
   - Large scale → Redis workers

2. **Limit concurrent workflows**:
   ```python
   # Don't create unlimited threads
   max_concurrent = min(len(items), os.cpu_count() * 2)
   ```

3. **Reuse workflow factories**:
   - Define factory once, call multiple times
   - Cache factory configurations

4. **Monitor resource usage**:
   - Track memory per workflow instance
   - Monitor thread count
   - Set timeouts

---

## Integration with Other Features

### With Distributed Execution

Combine factories with Redis workers:

```python
pipeline = create_etl_pipeline("distributed_etl")
pipeline.execute(
    "load",
    queue_backend=QueueBackend.REDIS,
    max_steps=1000
)
```

### With Dynamic Tasks

Use factories with runtime task generation:

```python
def create_dynamic_pipeline(name: str):
    ctx = workflow(name)

    with ctx:
        @task(inject_context=True)
        def dynamic_router(context):
            # Create tasks at runtime
            task = TaskWrapper("processor", lambda: process())
            context.next_task(task)

    return ctx
```

---

## Next Steps

After mastering workflow composition, explore:

1. **Real-World Examples** (`../09_real_world/`)
   - Apply composition patterns to production use cases
   - See complete implementations

2. **Visualization** (`../10_visualization/`)
   - Visualize composed workflows
   - Document complex patterns

3. **Custom Development**
   - Build your own workflow factories
   - Create domain-specific composition patterns

---

## Additional Resources

- **Python Threading**: https://docs.python.org/3/library/threading.html
- **concurrent.futures**: https://docs.python.org/3/library/concurrent.futures.html
- **Factory Pattern**: https://refactoring.guru/design-patterns/factory-method

---

**Ready to compose workflows?** Start with `workflow_factory.py`!
