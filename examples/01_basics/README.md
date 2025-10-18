# 01_basics - Getting Started with Graflow

This directory contains a minimal introduction to Graflow.

## Prerequisites

- Python 3.11+
- Graflow installed (`pip install -e .` from the project root)

## Example

### hello_world.py
**Concepts**: @task decorator, basic task execution

The simplest possible Graflow task. Learn how to:
- Define a task using the `@task` decorator
- Execute a task directly
- Understand the basic task execution model

```bash
python examples/01_basics/hello_world.py
```

## Important: Start with 02_workflows/

**This basic example is just an introduction.** For real Graflow usage, you should start with the examples in `02_workflows/`, which demonstrate:

- **Workflow context** (`workflow()` context manager)
- **Operators** (`>>` for sequential, `|` for parallel execution)
- **Context injection** (accessing `ExecutionContext`)
- **Proper task composition patterns**

The standalone `@task` decorator shown here is mainly used within workflow contexts, not in isolation.

## Next Steps

**→ Go directly to `examples/02_workflows/`** to learn the recommended workflow patterns:
1. **02_workflows/simple_pipeline.py** - Sequential task execution with `>>`
2. **02_workflows/operators_demo.py** - Parallel and mixed execution patterns
3. **02_workflows/context_injection.py** - Accessing workflow state
4. **02_workflows/workflow_decorator.py** - Reusable workflow definitions

After mastering workflows, explore:
- **03_data_flow/** - Channels for inter-task communication
- **04_execution/** - Custom execution handlers
- **05_distributed/** - Redis-based distributed execution

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
