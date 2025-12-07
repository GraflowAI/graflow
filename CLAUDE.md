# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graflow is an executable task graph engine for Python workflow execution. It provides both local and distributed execution capabilities with support for task graphs, parallel execution, inter-task communication via channels, cycle detection, and dynamic task generation.

**Tech Stack**: Python 3.11+, networkx, matplotlib, redis, cloudpickle
**Package Manager**: uv
**Dev Tools**: ruff (linting/formatting), mypy (type checking), pytest (testing)

## Essential Commands

### Development Workflow
```bash
# Setup
uv sync --dev              # Install development dependencies
uv sync --all-extras       # Install with graphviz extras (macOS: see install-extras target)

# Code Quality (run before commits)
make format                # Format code with ruff (auto-fix imports, whitespace)
make lint                  # Run ruff + mypy type checking
make test                  # Run pytest test suite
make check-all             # Run format + lint + test (full validation)
make fix                   # Quick fix with ruff --fix --unsafe-fixes

# Running Examples
make py examples/01_basics/hello_world.py    # Run specific example
PYTHONPATH=. uv run python <file>            # Alternative: run any Python file

# Testing
uvx pytest tests/ -v                         # Run all tests
uvx pytest tests/core/ -v                    # Run specific test directory
uvx pytest tests/core/test_task.py -v        # Run specific test file
uvx pytest tests/core/test_task.py::test_name -v  # Run specific test

# Single Test Pattern
PYTHONPATH=. python3 tests/worker/test_task_worker_integration.py  # Direct test execution
```

### Makefile Notes
- Line 23: Coverage reports to `flowlet` (legacy name, but works)
- Line 46-48: macOS graphviz extras require CFLAGS/LDFLAGS from Homebrew

## Architecture Overview

### Core Execution Model
Graflow uses a **task graph execution engine** where tasks are nodes and dependencies are edges. The engine supports:
- **Local execution**: In-process task execution
- **Distributed execution**: Redis-based queue with worker processes
- **Parallel execution**: `ParallelGroup` for concurrent task execution
- **Dynamic tasks**: Runtime task generation via `next_task()` and `next_iteration()`

### Key Components

#### Core Module (`graflow/core/`)
- **`engine.py`** - `WorkflowEngine`: Main execution engine with handler registry
  - Executes task graphs topologically
  - Supports custom handlers for different execution modes (direct, docker, etc.)
  - Methods: `execute()`, `execute_with_cycles()`, `register_handler()`

- **`task.py`** - Task abstraction layer
  - `Executable`: Base interface for all executable units
  - `Task`: Single executable task
  - `ParallelGroup`: Executes tasks concurrently
  - `TaskWrapper`: Wraps functions as tasks

- **`context.py`** - Execution state management
  - `ExecutionContext`: Global workflow execution state (channels, results, queue, graph)
  - `TaskExecutionContext`: Per-task execution context
  - `create_execution_context()`: Factory for context creation

- **`workflow.py`** - High-level workflow API
  - `workflow()`: Context manager for workflow definition
  - `WorkflowContext`: Manages task graph construction
  - Supports `>>` (sequential) and `|` (parallel) operators

- **`decorators.py`** - Function-to-task conversion
  - `@task`: Decorator to convert functions to Task objects
  - Integrates with serialization module for distributed execution

- **`checkpoint.py`** - Checkpoint/Resume functionality
  - `CheckpointManager`: Create and resume from checkpoints
  - `CheckpointState`: Execution state persistence
  - `CheckpointMetadata`: Checkpoint metadata management

- **`handler.py`** - Task handler interface
  - `TaskHandler`: Base class for custom execution strategies
  - Handler registration and dispatch system

- **`graph.py`** - Task graph (DAG) management
  - Uses networkx for graph operations
  - Cycle detection and topological sorting

#### Queue System (`graflow/queue/`)
- **`base.py`**: `TaskQueue` interface, `TaskSpec`, `TaskStatus`
- **`memory.py`**: `MemoryTaskQueue` - In-memory implementation
- **`redis.py`**: `RedisTaskQueue` - Distributed queue via Redis
- **`factory.py`**: `TaskQueueFactory` - Backend selection (memory/redis)

#### Channels (`graflow/channels/`)
- **`base.py`**: `Channel` interface for inter-task communication
- **`memory.py`**: `MemoryChannel` - Local state sharing
- **`redis.py`**: `RedisChannel` - Distributed state sharing
- **`typed.py`**: `TypedChannel` - Type-safe wrapper
- **`factory.py`**: `ChannelFactory` - Backend selection

#### Workers (`graflow/worker/`)
- **`worker.py`**: `TaskWorker` - Distributed task execution
- **`handler.py`**: `TaskHandler` interface
- **`main.py`**: Worker entry point (`python -m graflow.worker.main`)

#### Coordination (`graflow/coordination/`)
- **`executor.py`**: `GroupExecutor` - Parallel group execution
- **`redis.py`**: `RedisCoordinator` - Distributed coordination
- **`multiprocessing.py`**: Multiprocessing coordination

#### HITL - Human-in-the-Loop (`graflow/hitl/`)
- **`manager.py`**: `FeedbackManager` - Manages human feedback requests
- **`types.py`**: Feedback types and configuration
- **`notification.py`**: User notification system (Slack, webhooks, console)

#### LLM Integration (`graflow/llm/`)
- **`client.py`**: LLM client integration
- **`agents.py`**: LLM agent management and registration

#### Tracing (`graflow/trace/`)
- **`tracer.py`**: Tracing hooks for observability
- **`langfuse.py`**: Langfuse integration for workflow tracing

#### API (`graflow/api/`)
- **REST API**: Workflow management and HITL feedback endpoints
- **Feedback API**: HTTP endpoints for human feedback submission

#### Serialization (`graflow/serialization/`)
- **Cloudpickle-based**: Task and function serialization for distributed execution

#### Debug (`graflow/debug/`)
- **Visualization tools**: Workflow graph visualization and debugging utilities

### Key Design Patterns

1. **Abstract Base Classes**: `TaskQueue`, `Channel`, `Executable` with concrete implementations
2. **Factory Pattern**: Backend selection via `TaskQueueFactory`, `ChannelFactory`
3. **Context Managers**: `workflow()` context for graph construction
4. **Dependency Injection**: `ExecutionContext` passed through execution stack
5. **Handler Registry**: `WorkflowEngine` dispatches to registered handlers by task type

### Task Definition Patterns

**Method 1: @task decorator**
```python
from graflow.core.decorators import task

@task
def process_data(x: int) -> int:
    return x * 2

result = process_data.run(x=5)  # Execute directly
```

**Method 2: Workflow context with operators**
```python
from graflow.core.workflow import workflow

with workflow("my_workflow") as wf:
    task_a >> task_b  # Sequential: b depends on a
    task_c | task_d   # Parallel: c and d execute concurrently
    (task_a | task_b) >> task_c  # Mixed: a,b parallel then c

    wf.execute()  # Run the workflow
```

**Method 3: ParallelGroup**
```python
from graflow.core.task import ParallelGroup

parallel = ParallelGroup([task1, task2, task3], name="parallel_group")
```

### Execution Context Access

Tasks can access `ExecutionContext` via context injection:
```python
@task
def my_task(ctx: ExecutionContext):
    # Access channels
    ctx.channel("my_channel").put("data")

    # Store results
    ctx.results["key"] = value

    # Access graph
    dependencies = ctx.graph.dependencies(current_task)
```

### Dynamic Task Generation

**Runtime generation** (most common):
```python
@task
def dynamic_task(ctx: ExecutionContext):
    if some_condition:
        ctx.next_task(new_task)  # Add task to graph at runtime
    else:
        ctx.next_iteration(retry_task)  # Re-execute task
```

**Compile-time generation** (workflow setup):
```python
with workflow("dynamic") as wf:
    tasks = [create_task(i) for i in range(10)]
    # Build graph dynamically
```

### Distributed Execution

**Setup**:
1. Start Redis: `docker run -p 6379:6379 redis:7.2`
2. Start workers: `python -m graflow.worker.main --worker-id worker-1`
3. Submit workflow with Redis backend:

```python
from graflow.queue.factory import QueueBackend, TaskQueueFactory
from graflow.core.context import create_execution_context

context = create_execution_context(
    queue_backend=QueueBackend.REDIS,
    channel_backend="redis",
    queue_name="my_workflow"
)
# Tasks submitted to Redis queue, workers process them
```

## Code Style & Conventions

### Formatting & Linting
- **Tool**: ruff for linting/formatting, mypy for types
- **Line Length**: 120 characters
- **Python Version**: 3.10+ (target), 3.11+ (required)
- **Import Sorting**: isort with `graflow` as known first-party

### Type Hints
- Required for all function definitions (`disallow_untyped_defs = true`)
- No implicit optional (`no_implicit_optional = true`)
- Avoid `Any` when possible

### Naming
- Classes: `PascalCase` (e.g., `TaskQueue`, `ExecutionContext`)
- Functions/methods: `snake_case` (e.g., `get_next_task`, `add_to_queue`)
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

### Documentation
- Docstrings for all public classes/methods
- Include Args, Returns, Raises sections
- Type hints are primary parameter documentation

## Testing Strategy

### Test Structure
```
tests/
    core/             # Core functionality tests
    channels/         # Channel implementation tests
    coordination/     # Coordination tests
    queue/            # Queue implementation tests
    worker/           # Worker tests
    integration/      # Integration tests
    scenario/         # End-to-end scenario tests
    unit/             # Unit tests
```

### Common Test Patterns
- Use `pytest` fixtures from `conftest.py`
- Mock external dependencies (Redis, Docker)
- Test both memory and Redis backends for queues/channels
- Integration tests for distributed scenarios

## Examples Guide

The `examples/` directory contains **production-ready examples** organized by complexity:

1. **01_basics/**: Basic task introduction (1 example - hello_world.py)
2. **02_workflows/**: Workflow context, operators (`>>`, `|`), context injection (4 examples) - **Start here for real usage**
3. **03_data_flow/**: Channels, typed channels, result storage (3 examples)
4. **04_execution/**: Custom handlers (direct, docker, custom) (3 examples)
5. **05_distributed/**: Redis-based distribution, workers (3 examples)
6. **06_advanced/**: Lambda tasks, serialization, nested workflows (4 examples)
7. **07_dynamic_tasks/**: Dynamic task generation patterns (2 examples)
8. **09_real_world/**: ETL, ML training, batch processing, data analysis (4 examples)
9. **10_visualization/**: Workflow visualization (ASCII, Mermaid, PNG) (2 examples)

**Quick Start**: Begin with `examples/01_basics/hello_world.py`, then `examples/02_workflows/simple_pipeline.py`

## Important Implementation Notes

### Task Execution Flow
1. `WorkflowEngine.execute()` gets topological order from graph
2. For each task, engine calls `_get_handler()` to find appropriate handler
3. Handler executes task via `_execute_task()`
4. Results stored in `ExecutionContext.results`
5. Channels used for inter-task communication

### ExecutionContext Management
- Created via `create_execution_context()` factory
- Passed through entire execution stack
- Contains: graph, queue, channels, results, metadata
- Thread-safe for parallel execution

### Cycle Detection
- `CycleController` in `graflow/core/cycle.py`
- `execute_with_cycles()` for workflows with cycles
- Max iterations to prevent infinite loops

### Parallel Execution
- `ParallelGroup` uses threading/multiprocessing
- `GroupExecutor` manages parallel execution
- Results merged back to main context

### Serialization
- Uses `cloudpickle` for function/task serialization
- Serialization module handles task and function serialization
- Critical for distributed execution across workers

### Checkpoint/Resume System
- **User-controlled checkpointing**: Tasks call `context.checkpoint()` to schedule checkpoints
- **Automatic state persistence**: Engine saves checkpoint after task completion
- **Three-file structure**: `.pkl` (context), `.state.json` (execution state), `.meta.json` (metadata)
- **Resume capability**: `CheckpointManager.resume_from_checkpoint(path)` restores full workflow state
- **Production-ready**: Supports local filesystem, Redis backend planned
- **Use cases**: Long-running ML training, multi-hour pipelines, fault tolerance

### Human-in-the-Loop (HITL)
- **Feedback requests**: Tasks can request human input during execution
- **Multiple feedback types**: Approval, text input, selection, multi-selection, custom
- **Intelligent timeout handling**: Short polling → checkpoint → resume on feedback
- **Universal notifications**: Console, Slack webhooks, Discord, Teams, custom endpoints
- **Distributed support**: Redis-based feedback persistence for cross-worker scenarios
- **Channel integration**: Automatic writing of feedback responses to channels
- **Use cases**: Deployment approvals, data validation, parameter tuning, error recovery

### Tracing and Observability
- **Tracer hooks**: `on_workflow_start()`, `on_workflow_end()`, `on_task_start()`, `on_task_end()`
- **Langfuse integration**: Automatic tracing to Langfuse platform
- **Trace ID propagation**: Distributed trace tracking across workers
- **LLM integration**: Track LLM calls and agent interactions
- **Use cases**: Debugging, performance monitoring, workflow analytics

## Common Development Tasks

### Adding a New Task Type
1. Extend `Executable` interface in `graflow/core/task.py`
2. Implement `execute(context: ExecutionContext)` method
3. Register handler in `WorkflowEngine` if needed
4. Add tests in `tests/core/`

### Adding a New Backend
1. Implement abstract interface (`TaskQueue` or `Channel`)
2. Add to factory (`TaskQueueFactory` or `ChannelFactory`)
3. Add enum value to `QueueBackend`
4. Add integration tests

### Debugging Workflows
1. Enable verbose logging in `WorkflowEngine`
2. Use `ctx.results` to inspect intermediate values
3. Visualize graph: `examples/10_visualization/workflow_visualization.py`
4. Check Redis queue state: `redis-cli LLEN queue_name`

### Running Distributed Workflows
1. Start Redis: `docker run -p 6379:6379 redis:7.2`
2. Start workers (multiple terminals):
   ```bash
   python -m graflow.worker.main --worker-id worker-1
   python -m graflow.worker.main --worker-id worker-2
   ```
3. Run workflow with Redis backend (see examples/05_distributed/)

## Known Issues & Gotchas

### Current State
- Git status shows modifications to examples and core files
- Recent work on parallel execution, context injection, and diamond workflows
- Active development in parallel execution design (see `docs/parallel_execution_*.md`)

### Parallel Execution
- Diamond workflows (A -> B,C -> D) require careful dependency tracking
- Context injection in nested iterations needs proper task ID handling
- See recent commits for parallel group fixes

### Testing
- Makefile line 23: Coverage target uses legacy name `flowlet` instead of `graflow`
- Integration tests may require Redis/Docker running
- Some tests use `PYTHONPATH=. python3` pattern for direct execution

## Project-Specific Vocabulary

- **Executable**: Base interface for anything that can be executed
- **Task**: Single unit of work
- **ParallelGroup**: Collection of tasks executed concurrently
- **TaskSpec**: Specification for a task (params, retry, status)
- **ExecutionContext**: Global workflow state
- **TaskExecutionContext**: Per-task execution state
- **Channel**: Inter-task communication primitive
- **TaskQueue**: Queue for distributed task execution
- **Handler**: Executor for specific task types (DirectTaskHandler, DockerTaskHandler)
- **next_task()**: Runtime task addition to graph
- **next_iteration()**: Runtime task re-execution
- **terminate_workflow()**: Early normal termination of workflow
- **cancel_workflow()**: Abnormal workflow cancellation with error
- **Checkpoint**: Snapshot of workflow execution state
- **FeedbackManager**: Manages HITL feedback requests and responses
- **Tracer**: Observability hooks for workflow/task execution tracking

## Git Workflow

Current branch: `main`
Recent focus: Parallel execution fixes, context injection, diamond workflow testing

Before committing:
```bash
make format     # Auto-fix formatting
make lint       # Check linting + types
make test       # Run test suite
```
