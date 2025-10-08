# Advanced Patterns

This directory contains advanced Graflow examples demonstrating sophisticated patterns and techniques for power users.

## Overview

These examples showcase advanced capabilities of Graflow including:
- **Dynamic Task Generation** - Creating tasks at runtime based on configuration
- **Lambda and Closure Tasks** - Using functional programming patterns
- **Custom Serialization** - Understanding cloudpickle for distributed execution
- **Nested Workflows** - Hierarchical workflow composition and organization

## Examples

### 1. lambda_tasks.py
**Difficulty**: Advanced
**Time**: 15 minutes

Demonstrates cloudpickle serialization for lambda functions and closures.

**Key Concepts**:
- Lambda task creation
- Closure tasks with captured state
- Factory pattern for task generation
- Serialization for distributed execution

**What You'll Learn**:
- How to create tasks from lambdas
- Capturing variables in closures
- When to use functional patterns
- Cloudpickle vs standard pickle

**Run**:
```bash
python examples/06_advanced/lambda_tasks.py
```

**Expected Output**:
```
=== Lambda and Closure Tasks ===

Step 1: Creating tasks from lambdas and closures
✅ Lambda task created
✅ Closure task created
✅ Complex closure task created

Step 2: Building workflow
✅ Task graph built

Step 3: Executing workflow
⏳ Executing tasks...
✅ lambda_task executed -> 42
✅ double_task executed -> 84
✅ multiply_by_10 executed -> 840
```

---

### 2. dynamic_tasks.py
**Difficulty**: Advanced
**Time**: 20 minutes

Shows how to dynamically generate and configure tasks at runtime.

**Key Concepts**:
- Runtime task creation
- Task factory patterns
- Conditional workflow composition
- Configuration-driven pipelines

**What You'll Learn**:
- Creating tasks in loops
- Building workflows from configuration
- Factory functions for tasks
- Dynamic DAG construction

**Run**:
```bash
python examples/06_advanced/dynamic_tasks.py
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
Configuration: enable_validation=True, enable_transform=True
Building conditional pipeline...
✅ Extract task added
✅ Validation task added (conditional)
```

---

### 3. custom_serialization.py
**Difficulty**: Advanced
**Time**: 15 minutes

Explores cloudpickle's serialization capabilities for complex Python objects.

**Key Concepts**:
- Cloudpickle vs standard pickle
- Serializing lambdas and closures
- Serializing class instances with state
- Nested closure serialization

**What You'll Learn**:
- What cloudpickle can serialize
- Limitations and pitfalls
- Best practices for distributed execution
- How to avoid serialization issues

**Run**:
```bash
python examples/06_advanced/custom_serialization.py
```

**Expected Output**:
```
=== Custom Serialization Demo ===

Test 1: Serializing Lambda Functions
✅ Lambda function serialized and deserialized
   Original result: 15
   Deserialized result: 15

Test 2: Serializing Closures
✅ Closure with captured state serialized
   Factory multiplier: 10
   Result: 50
```

---

### 4. nested_workflow.py
**Difficulty**: Advanced
**Time**: 20 minutes

Demonstrates nested workflow contexts for hierarchical organization.

**Key Concepts**:
- Nested workflow contexts
- Inner and outer workflow scoping
- Workflow composition and reusability
- Hierarchical task organization
- Modular workflow design

**What You'll Learn**:
- How to nest workflows within workflows
- Organizing complex pipelines hierarchically
- Creating reusable workflow components
- Managing workflow scope and isolation

**Run**:
```bash
python examples/06_advanced/nested_workflow.py
```

**Expected Output**:
```
=== Nested Workflows Demo ===

Scenario 1: Basic Nested Workflows
📦 Outer Workflow: data_processing
   🔹 Inner Workflow: validation_workflow
      ✅ validate_schema completed
      ✅ validate_values completed
   ✅ Inner validation workflow completed

   🔹 Inner Workflow: transformation_workflow
      ✅ normalize_data completed
      ✅ enrich_data completed
   ✅ Inner transformation workflow completed

✅ Outer workflow completed
```

---

## Learning Path

**Recommended Order**:
1. Start with `lambda_tasks.py` to understand functional patterns
2. Move to `dynamic_tasks.py` for runtime task creation
3. Explore `nested_workflow.py` for hierarchical organization
4. Finish with `custom_serialization.py` for deep understanding

**Prerequisites**:
- Complete examples from 01-04
- Understanding of Python closures
- Familiarity with functional programming concepts

**Total Time**: ~70 minutes

---

## Key Concepts

### Dynamic Task Creation

Create tasks at runtime based on configuration:

```python
@task
def create_processor(config):
    def processor():
        # Use config
        return process_with_config(data, config)
    return processor

# Create tasks dynamically
for config in configs:
    task = create_processor(config)
    tasks.append(task)
```

### Lambda Tasks

Use lambda functions for simple inline tasks:

```python
from graflow.core.task import TaskWrapper

lambda_task = TaskWrapper("add_ten", lambda x: x + 10)
```

### Closures with State

Factory functions that capture state:

```python
def create_multiplier(factor):
    @task
    def multiply(x):
        return x * factor  # Captures 'factor'
    return multiply

multiply_by_10 = create_multiplier(10)
```

### Cloudpickle Serialization

Understanding what can be serialized:

**Can serialize** ✅:
- Lambda functions
- Closures with captured variables
- Nested functions
- Class instances with state
- Most Python objects

**Cannot serialize** ❌:
- File handles
- Database connections
- Network sockets
- Thread/process objects
- Some C extensions

---

## Common Patterns

### 1. Configuration-Driven Workflows

Build workflows from configuration files:

```python
config = load_config("workflow.yaml")

with workflow("dynamic") as ctx:
    tasks = []
    for step_config in config["steps"]:
        if step_config["enabled"]:
            task = create_task_from_config(step_config)
            tasks.append(task)

    # Chain tasks
    for i in range(len(tasks) - 1):
        tasks[i] >> tasks[i + 1]
```

### 2. Batch Processing with Dynamic Tasks

Process variable numbers of items:

```python
num_items = len(data)
batch_size = 100
num_batches = (num_items + batch_size - 1) // batch_size

with workflow("batch") as ctx:
    batch_tasks = []
    for i in range(num_batches):
        @task(id=f"batch_{i}")
        def process_batch():
            start = i * batch_size
            end = min(start + batch_size, num_items)
            return process_items(data[start:end])

        batch_tasks.append(process_batch)

    # Aggregate results
    @task
    def aggregate():
        results = [ctx.get_result(f"batch_{i}")
                  for i in range(num_batches)]
        return combine(results)

    for batch_task in batch_tasks:
        batch_task >> aggregate
```

### 3. Task Factory Pattern

Reusable task generators:

```python
def create_api_caller(endpoint, method="GET"):
    """Factory for API calling tasks."""
    @task(id=f"call_{endpoint.replace('/', '_')}")
    def call_api():
        response = requests.request(method, endpoint)
        return response.json()
    return call_api

# Use factory
get_users = create_api_caller("/api/users")
get_posts = create_api_caller("/api/posts")
```

---

## Best Practices

### ✅ DO:

- Use unique task IDs for dynamically created tasks
- Keep closures lightweight
- Test serialization before distributing
- Document dynamic behavior
- Use factory functions for reusable patterns

### ❌ DON'T:

- Capture large objects in closures
- Capture non-serializable objects (file handles, connections)
- Create tasks with duplicate IDs
- Over-complicate with dynamic creation when static works
- Forget to clean up resources in dynamic tasks

---

## Troubleshooting

### DuplicateTaskError

**Problem**: Multiple tasks with the same ID

**Solution**: Use unique IDs for dynamic tasks:
```python
@task(id=f"task_{i}")  # Include loop variable
def my_task():
    pass
```

### PickleError / SerializationError

**Problem**: Object cannot be serialized

**Solution**: Avoid capturing non-serializable objects:
```python
# Bad
db_conn = connect_db()
task = lambda: query(db_conn)  # db_conn not serializable

# Good
def create_task(connection_string):
    def task():
        conn = connect_db(connection_string)
        result = query(conn)
        conn.close()
        return result
    return task
```

### Memory Issues with Large Closures

**Problem**: Closure captures large dataset

**Solution**: Pass data path instead:
```python
# Bad
large_data = load_huge_dataset()
task = lambda: process(large_data)  # Captures all data

# Good
def create_task(data_path):
    def task():
        data = load_dataset(data_path)  # Load in task
        return process(data)
    return task
```

---

## Real-World Use Cases

### 1. A/B Testing Workflows

Create different pipeline variants:

```python
variants = ["variant_a", "variant_b", "variant_c"]

for variant in variants:
    @task(id=f"test_{variant}")
    def run_variant():
        config = load_variant_config(variant)
        return execute_experiment(config)
```

### 2. Multi-Tenant Processing

Process data for multiple tenants:

```python
tenants = get_active_tenants()

for tenant_id in tenants:
    @task(id=f"process_{tenant_id}")
    def process_tenant():
        data = fetch_tenant_data(tenant_id)
        return process_data(data)
```

### 3. Feature Flag Driven Pipelines

Enable/disable features dynamically:

```python
features = load_feature_flags()

with workflow("conditional") as ctx:
    extract_task = extract_data()
    last_task = extract_task

    if features["enable_validation"]:
        validate = validate_data()
        last_task >> validate
        last_task = validate

    if features["enable_enrichment"]:
        enrich = enrich_data()
        last_task >> enrich
        last_task = enrich
```

---

## Performance Considerations

### Task Creation Overhead

Dynamic task creation has minimal overhead:
- Task object creation: ~1-5ms
- Registration in graph: ~1-2ms
- Total per task: ~2-7ms

For 1000 dynamic tasks: ~2-7 seconds overhead

### Serialization Performance

Cloudpickle serialization times:
- Simple function: <1ms
- Closure with small state: 1-5ms
- Complex nested closure: 5-20ms
- Class instance: 10-50ms

### Memory Usage

Each task object uses approximately:
- Base task: ~1KB
- With captured state: +size of captured objects
- 1000 tasks with minimal state: ~1-5MB

---

## Advanced Topics

### Recursive Task Generation

Create tasks recursively:

```python
def create_recursive_tasks(depth, max_depth=5):
    @task(id=f"level_{depth}")
    def task_at_depth():
        result = process_level(depth)
        if depth < max_depth:
            # Trigger next level
            return result
        return result

    if depth < max_depth:
        next_task = create_recursive_tasks(depth + 1, max_depth)
        task_at_depth >> next_task

    return task_at_depth
```

### Parameterized Task Classes

Use classes for complex task state:

```python
class ConfigurableTask:
    def __init__(self, config):
        self.config = config

    @task
    def execute(self):
        return self.process(self.config)

    def process(self, config):
        # Implementation
        pass

# Create tasks from class
tasks = [ConfigurableTask(cfg).execute() for cfg in configs]
```

### Memoization with Closures

Cache results in closure:

```python
def create_memoized_task(expensive_function):
    cache = {}

    @task
    def memoized(*args):
        key = str(args)
        if key not in cache:
            cache[key] = expensive_function(*args)
        return cache[key]

    return memoized
```

---

## Integration with Distributed Execution

These patterns work seamlessly with Redis-based distribution:

```python
# Dynamic tasks are serialized and distributed
with workflow("distributed_dynamic") as ctx:
    # Create 100 tasks dynamically
    tasks = []
    for i in range(100):
        @task(id=f"worker_task_{i}")
        def process():
            return heavy_computation(i)
        tasks.append(process)

    # Execute with Redis backend
    ctx.execute(
        start_node="worker_task_0",
        queue_backend=QueueBackend.REDIS
    )
```

All tasks will be:
1. Serialized with cloudpickle
2. Sent to Redis queue
3. Picked up by workers
4. Deserialized and executed
5. Results returned via Redis

---

## Next Steps

After mastering these advanced patterns, explore:

1. **Distributed Execution** (`../05_distributed/`)
   - Scale dynamic workflows across workers
   - Use Redis for task distribution

2. **Real-World Examples** (`../07_real_world/`)
   - Apply patterns to production use cases
   - See complete implementations

3. **Custom Development**
   - Build your own task factories
   - Create domain-specific workflow generators

---

## Additional Resources

- **Cloudpickle Documentation**: https://github.com/cloudpipe/cloudpickle
- **Python Closures**: https://docs.python.org/3/reference/datamodel.html#index-34
- **Factory Pattern**: https://refactoring.guru/design-patterns/factory-method

---

**Ready to apply these patterns?** Check out the real-world examples in `07_real_world/`!
