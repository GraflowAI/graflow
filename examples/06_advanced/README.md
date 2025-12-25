# Advanced Patterns

This directory contains advanced Graflow examples demonstrating sophisticated patterns and techniques for power users.

## Overview

These examples showcase advanced capabilities of Graflow including:
- **Lambda and Closure Tasks** - Using functional programming patterns
- **Custom Serialization** - Understanding cloudpickle for distributed execution
- **Nested Workflows** - Hierarchical workflow composition and organization
- **Context Management** - Global and explicit context handling

**Note**: For dynamic task generation examples, see:
- **Dynamic Tasks** → `../07_dynamic_tasks/`

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
uv run python examples/06_advanced/lambda_tasks.py
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

**Real-World Applications**:
- Quick prototyping of workflows
- Functional data transformations
- Factory-based task generation
- Distributed processing with simple functions

---

### 2. custom_serialization.py
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
uv run python examples/06_advanced/custom_serialization.py
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

**Real-World Applications**:
- Distributed task execution
- Saving workflow state
- Task migration across workers
- Understanding serialization limits

---

### 3. nested_workflow.py
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
uv run python examples/06_advanced/nested_workflow.py
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

**Real-World Applications**:
- Modular data pipelines
- Reusable workflow components
- Complex multi-stage processing
- Hierarchical workflow organization

---

### 4. global_context.py
**Difficulty**: Advanced
**Time**: 20 minutes

Explores global workflow context behavior and fallback mechanisms.

**Key Concepts**:
- Global context auto-creation
- Explicit vs implicit contexts
- Context isolation
- current_workflow_context() usage

**What You'll Learn**:
- How global context works
- When to use explicit contexts
- Context isolation patterns
- Best practices for context management

**Run**:
```bash
uv run python examples/06_advanced/global_context.py
```

**Expected Output**:
```
=== Global Workflow Context ===

Creating tasks without explicit context...
✅ Global tasks created

Global workflow context:
...
✅ Global context automatically created
```

**Real-World Applications**:
- Quick prototyping
- Simple scripts without explicit contexts
- Understanding context behavior
- Debugging workflow issues

---

## Learning Path

**Recommended Order**:
1. Start with `lambda_tasks.py` to understand functional patterns
2. Learn `custom_serialization.py` for deep understanding
3. Explore `nested_workflow.py` for hierarchical organization
4. Understand `global_context.py` for context management

**Prerequisites**:
- Complete examples from 01-04
- Understanding of Python closures
- Familiarity with functional programming concepts

**Total Time**: ~70 minutes (~1 hour)

---

## Key Concepts

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

### Task Factory Pattern

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

### Nested Workflow Organization

Organize complex workflows hierarchically:

```python
with workflow("outer") as outer_ctx:
    @task
    def run_validation():
        with workflow("validation") as val_ctx:
            # Inner workflow tasks
            validate_schema()
            validate_values()
            val_ctx.execute("validate_schema")

    @task
    def run_transformation():
        with workflow("transformation") as trans_ctx:
            # Inner workflow tasks
            normalize()
            enrich()
            trans_ctx.execute("normalize")

    run_validation >> run_transformation
```

---

## Best Practices

### ✅ DO:

- Keep closures lightweight
- Test serialization before distributing
- Document captured variables
- Use factory functions for reusable patterns
- Prefer explicit contexts for clarity

### ❌ DON'T:

- Capture large objects in closures
- Capture non-serializable objects (file handles, connections)
- Over-complicate with lambdas when named functions are clearer
- Forget to clean up resources in closures
- Rely on global context in production code

---

## Troubleshooting

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

### Context Confusion

**Problem**: Task registered in wrong context

**Solution**: Use explicit contexts:
```python
# Bad - uses global context
@task
def my_task():
    pass

# Good - explicit context
with workflow("my_workflow") as ctx:
    @task
    def my_task():
        pass
```

---

## Real-World Use Cases

### 1. Functional Data Transformations

Quick transformations with lambdas:

```python
# Create transformation pipeline
transform1 = TaskWrapper("normalize", lambda x: x / max_value)
transform2 = TaskWrapper("scale", lambda x: x * 100)
transform3 = TaskWrapper("round", lambda x: round(x, 2))

# Chain transformations
transform1 >> transform2 >> transform3
```

### 2. Parameterized Processing

Factory pattern for configurable tasks:

```python
def create_processor(config):
    """Create processor task from configuration."""
    @task(id=f"process_{config['id']}")
    def process():
        data = load_data(config['source'])
        result = apply_rules(data, config['rules'])
        save_result(result, config['destination'])
        return result
    return process

# Create multiple processors
processors = [create_processor(cfg) for cfg in configs]
```

### 3. Modular Pipeline Components

Reusable workflow modules:

```python
def create_validation_module(name):
    """Reusable validation workflow component."""
    with workflow(name) as ctx:
        @task
        def validate_schema():
            # Schema validation logic
            pass

        @task
        def validate_business_rules():
            # Business rule validation
            pass

        validate_schema >> validate_business_rules
        return ctx

# Use in larger pipeline
with workflow("main") as main_ctx:
    @task
    def run_validation():
        val_ctx = create_validation_module("validation")
        val_ctx.execute("validate_schema")
```

---

## Performance Considerations

### Task Creation Overhead

Lambda and closure task creation:
- Lambda task: ~1-2ms
- Closure with small state: ~2-5ms
- Complex nested closure: ~5-15ms

### Serialization Performance

Cloudpickle serialization times:
- Simple lambda: <1ms
- Closure with small state: 1-5ms
- Complex nested closure: 5-20ms
- Class instance: 10-50ms

### Memory Usage

Each task object uses approximately:
- Lambda task: ~0.5-1KB
- Closure task: ~1-2KB + captured state size
- Nested workflow: ~2-5KB per level

---

## Integration with Distributed Execution

These patterns work seamlessly with Redis-based distribution:

```python
# Lambda tasks are serialized and distributed
with workflow("distributed") as ctx:
    # Create task with closure
    factor = 10
    multiply_task = TaskWrapper("multiply", lambda x: x * factor)

    # Execute with Redis backend
    ctx.execute(
        start_node="multiply",
        queue_backend=QueueBackend.REDIS
    )
```

The task will be:
1. Serialized with cloudpickle
2. Sent to Redis queue
3. Picked up by workers
4. Deserialized and executed
5. Results returned via Redis

---

## Next Steps

After mastering these advanced patterns, explore:

1. **Dynamic Tasks** (`../07_dynamic_tasks/`)
   - Compile-time and runtime task generation
   - Iterative processing patterns
   - State machines

2. **Real-World Examples** (`../09_real_world/`)
   - Apply patterns to production use cases
   - See complete implementations

---

## Additional Resources

- **Cloudpickle Documentation**: https://github.com/cloudpipe/cloudpickle
- **Python Closures**: https://docs.python.org/3/reference/datamodel.html#index-34
- **Factory Pattern**: https://refactoring.guru/design-patterns/factory-method

---

**Ready for more advanced patterns?** Check out dynamic tasks in `07_dynamic_tasks/`!
