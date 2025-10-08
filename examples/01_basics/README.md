# 01_basics - Getting Started with Graflow

This directory contains the most basic Graflow examples to help you get started.

## Prerequisites

- Python 3.11+
- Graflow installed (`pip install -e .` from the project root)

## Examples

### 1. hello_world.py
**Concepts**: @task decorator, basic task execution

The simplest possible Graflow workflow. Learn how to:
- Define a task using the `@task` decorator
- Execute a task
- Understand the basic task execution model

```bash
python examples/01_basics/hello_world.py
```

### 2. task_dependencies.py
**Concepts**: Data flow, task composition

Learn how to:
- Pass data between tasks using return values
- Create a simple data processing pipeline
- Chain tasks together

```bash
python examples/01_basics/task_dependencies.py
```

### 3. task_with_parameters.py
**Concepts**: Task parameters, type hints, nested execution

Learn how to:
- Define tasks with parameters
- Use type hints for clarity
- Chain and nest task execution
- Build flexible, reusable tasks

```bash
python examples/01_basics/task_with_parameters.py
```

## Key Concepts

### Tasks
Tasks are the fundamental building blocks in Graflow. A task is simply a Python function decorated with `@task`:

```python
from graflow.core.decorators import task

@task
def my_task():
    return "Hello, Graflow!"
```

### Calling Tasks
Tasks can be called just like regular functions:

```python
result = my_task()  # Execute the task
print(result)  # "Hello, Graflow!"
```

### Passing Data
Tasks can accept parameters and return values:

```python
@task
def process(data):
    return data * 2

result = process(21)  # 42
```

## Next Steps

Once you're comfortable with these basics, move on to:
- **02_workflows/** - Learn about workflow orchestration
- **03_data_flow/** - Explore channels for inter-task communication
- **04_execution/** - Custom execution handlers

## Troubleshooting

### Import Errors
If you get import errors, make sure Graflow is installed:
```bash
cd /path/to/graflow
pip install -e .
```

### Task Not Executing
Make sure you're calling the task (with parentheses):
```python
result = my_task()  # Correct
result = my_task    # Wrong - just assigns the task object
```

## API Reference

- `@task` - Decorator to create a task from a function
- `@task(id="custom_id")` - Create a task with a custom ID
- `@task(inject_context=True)` - Inject ExecutionContext into the task

For more details, see the [main documentation](../../docs/).
