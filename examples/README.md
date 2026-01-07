# Graflow Examples

Welcome to the Graflow examples! This directory contains progressive examples to help you learn Graflow from basics to advanced use cases.

## 🎉 What's Available

**47 comprehensive, production-ready examples** covering:
- ✅ **Task Basics** - Define and execute tasks with parameters
- ✅ **Workflow Orchestration** - Sequential and parallel task composition
- ✅ **Data Flow** - Channels, typed communication, and result storage
- ✅ **Execution Control** - Direct, Docker, and custom handlers
- ✅ **Distributed Execution** - Redis-based task distribution across workers
- ✅ **Advanced Patterns** - Dynamic tasks, lambdas, and custom serialization
- ✅ **Real-World Use Cases** - Production-ready ETL, ML, and batch processing
- ✅ **Workflow Visualization** - ASCII, Mermaid, and PNG graph visualizations
- ✅ **Group Execution Policies** - Flexible error handling for parallel tasks
- ✅ **LLM Integration** - AI-powered workflows with LLMClient and agents
- ✅ **Human-in-the-Loop** - Interactive workflows with human feedback
- ✅ **Checkpoint/Resume** - Workflow state persistence and fault tolerance

All examples include detailed documentation, real-world use cases, and hands-on experiments!

## Quick Start

```bash
# Install Graflow
cd /path/to/graflow
uv sync --dev

# Run your first example
uv run python examples/01_basics/hello_world.py
```

## Example Categories

### ✅ 01_basics - Getting Started
**Status**: Complete | **Difficulty**: Beginner

Learn the fundamentals of Graflow:
- Defining and executing tasks
- Passing data between tasks
- Working with parameters

[View basics examples →](01_basics/)

### ✅ 02_workflows - Workflow Orchestration
**Status**: Complete | **Difficulty**: Intermediate

Master workflow orchestration with context managers, operators, and dependency management:
- Creating and executing workflows
- Sequential (`>>`) and parallel (`|`) operators
- Accessing execution context within tasks

[View workflow examples →](02_workflows/)

### ✅ 03_data_flow - Inter-Task Communication
**Status**: Complete | **Difficulty**: Intermediate

Learn how tasks communicate and share data through channels and results:
- Basic channel operations
- Type-safe channels with TypedDict
- Task result storage and retrieval

[View data flow examples →](03_data_flow/)

### ✅ 04_execution - Custom Execution Handlers
**Status**: Complete | **Difficulty**: Advanced

Control task execution with custom handlers and isolated environments:
- Direct (in-process) execution
- Docker container execution
- Building custom handlers

[View execution examples →](04_execution/)

### ✅ 05_distributed - Distributed Execution
**Status**: Complete | **Difficulty**: Advanced

Scale workflows across multiple workers using Redis:
- Redis-based task queues
- Worker process management
- Distributed workflow coordination

[View distributed examples →](05_distributed/)

**Note**: Requires Redis server. See directory README for setup instructions.

### ✅ 06_advanced - Advanced Patterns
**Status**: Complete | **Difficulty**: Expert

Advanced workflow patterns:
- Lambda and closure tasks
- Custom serialization with cloudpickle
- Nested workflow composition
- Global context management
- Modular task organization (Extract/Transform/Load pattern)

[View advanced examples →](06_advanced/)

### ✅ 07_dynamic_tasks - Dynamic Task Generation
**Status**: Complete | **Difficulty**: Advanced

Dynamic task generation patterns:
- Compile-time dynamic task creation
- Runtime task generation with next_task() and next_iteration()
- Iterative processing and convergence patterns
- State machines and conditional branching

[View dynamic task examples →](07_dynamic_tasks/)

### ✅ 08_real_world - Real-World Use Cases
**Status**: Complete | **Difficulty**: Intermediate to Advanced

Complete production-ready examples:
- ETL data pipeline with validation
- Machine learning training workflow
- Batch processing for large datasets
- Sales data analysis with anomaly detection

[View real-world examples →](08_real_world/)

### ✅ 09_visualization - Workflow Visualization
**Status**: Complete | **Difficulty**: Intermediate

Visualize workflows and graphs in multiple formats:
- Workflow graph extraction from TaskGraph
- ASCII visualization for terminal output
- Mermaid diagram generation for documentation
- PNG generation for presentations and reports
- Graph analysis and dependency visualization

[View visualization examples →](09_visualization/)

**Note**: Some features require optional dependencies (grandalf, pygraphviz). See directory README for details.

### ✅ 10_group_exec_policy - Parallel Group Error Handling
**Status**: Complete | **Difficulty**: Advanced

Master error handling strategies for parallel task execution:
- Strict mode (all tasks must succeed)
- Best-effort execution (continue despite failures)
- At-least-N policy (minimum success threshold)
- Critical tasks policy (only specific tasks must succeed)
- Custom policy implementation

[View group execution policy examples →](10_group_exec_policy/)

### ✅ 11_llm_integration - LLM Integration
**Status**: Complete | **Difficulty**: Intermediate to Advanced

Build AI-powered workflows with LLM integration:
- LLMClient injection for direct LLM API access (OpenAI, Anthropic, Google, etc.)
- Per-task model override for cost/performance optimization
- LLMAgent injection with Google ADK for ReAct/Supervisor patterns
- Multi-agent workflows with specialized agents
- Unified tracing with Langfuse integration

[View LLM integration examples →](11_llm_integration/)

**Note**: Requires LiteLLM (`uv add litellm`). Agent examples require Google ADK (`uv add google-adk`).

### ✅ 12_hitl - Human-in-the-Loop
**Status**: Complete | **Difficulty**: Intermediate to Advanced

Build interactive workflows with human feedback:
- Basic approval workflows with immediate feedback
- Timeout handling and checkpoint creation
- Channel integration for feedback sharing
- REST API for external feedback submission
- Distributed HITL with Redis backend

[View HITL examples →](12_hitl/)

### ✅ 13_checkpoints - Checkpoint/Resume
**Status**: Complete | **Difficulty**: Intermediate to Advanced

Master workflow state persistence and fault tolerance:
- Basic checkpoint creation and resumption
- State machine workflows with checkpoint at each transition
- Periodic checkpoints for long-running tasks
- Fault recovery with automatic retry
- Production-ready patterns for ML training and data pipelines

[View checkpoint examples →](13_checkpoints/)

**Use Cases**: Long-running ML training, multi-hour data pipelines, fault-tolerant workflows, state machine workflows

## Learning Path

**Recommended order for beginners:**

### Level 1: Fundamentals (Start Here!) 🌱

**01_basics/** - Master task creation and execution
1. `hello_world.py` - Your first Graflow task (5 min)
2. `task_dependencies.py` - Data flow between tasks (10 min)
3. `task_with_parameters.py` - Flexible task execution (10 min)

### Level 2: Orchestration 🔀

**02_workflows/** - Learn workflow composition
1. `simple_pipeline.py` - The simplest workflow (5 min) ⭐ START HERE
2. `workflow_decorator.py` - Using the workflow context manager (15 min)
3. `operators_demo.py` - Sequential and parallel execution (15 min)
4. `context_injection.py` - Accessing execution context (15 min)
5. `task_graph_lowlevel_api.py` - Low-level TaskGraph API usage (20 min)

### Level 3: Communication 📡

**03_data_flow/** - Master inter-task communication
1. `channels_basic.py` - Channel operations (15 min)
2. `typed_channels.py` - Type-safe channels with TypedDict (15 min)
3. `results_storage.py` - Task result storage and retrieval (15 min)

### Level 4: Advanced Control 🎛️

**04_execution/** - Control task execution
1. `direct_handler.py` - Direct in-process execution (10 min)
2. `docker_handler.py` - Docker container execution (20 min)
3. `custom_handler.py` - Building custom handlers (20 min)

### Level 5: Distributed Systems 🌐

**05_distributed/** - Scale across multiple workers
1. `redis_basics.py` - Redis integration basics (15 min)
2. `redis_worker.py` - Worker execution pattern (15 min)
3. `distributed_workflow.py` - Complete distributed pipeline (20 min)

### Level 6: Advanced Patterns 🚀

**06_advanced/** - Expert-level techniques
1. `lambda_tasks.py` - Functional programming patterns (15 min)
2. `custom_serialization.py` - Understanding cloudpickle (15 min)
3. `nested_workflow.py` - Hierarchical workflow organization (20 min)
4. `global_context.py` - Context management patterns (20 min)
5. `modular_etl.py` - Modular task organization with separate files (20 min)

### Level 7: Dynamic Task Generation 🎯

**07_dynamic_tasks/** - Dynamic task creation patterns
1. `dynamic_tasks.py` - Compile-time task generation (20 min)
2. `runtime_dynamic_tasks.py` - Runtime task creation with next_task() (30 min)

### Level 8: Production Use Cases 💼

**08_real_world/** - Complete real-world applications
1. `data_pipeline.py` - ETL workflow (20 min)
2. `ml_training.py` - ML training pipeline (20 min)
3. `batch_processing.py` - Large-scale batch processing (15 min)
4. `sales_analysis.py` - Data analysis with anomaly detection (25 min)

### Level 9: Workflow Visualization 📊

**09_visualization/** - Visualize workflows and graphs
1. `workflow_visualization.py` - Visualizing Graflow workflows (20 min)
2. `graph_utilities.py` - Graph visualization utilities (25 min)

### Level 10: Group Execution Policy ⚖️

**10_group_exec_policy/** - Parallel group error handling strategies
1. `parallel_group_strict_mode.py` - All tasks must succeed (15 min)
2. `parallel_group_best_effort.py` - Continue even if some fail (15 min)
3. `parallel_group_at_least_n.py` - Require minimum successes (20 min)
4. `parallel_group_critical_tasks.py` - Only critical tasks must succeed (20 min)
5. `parallel_group_custom_policy.py` - Build custom policies (25 min)

### Level 11: LLM Integration 🤖

**11_llm_integration/** - AI-powered workflows
1. `simple_llm_client.py` - Basic LLMClient injection (15 min)
2. `model_override.py` - Cost optimization with model selection (20 min)
3. `llm_agent.py` - ReAct patterns with Google ADK (25 min)
4. `multi_agent_workflow.py` - Multi-agent collaboration (30 min)

### Level 12: Human-in-the-Loop (HITL) 👤

**12_hitl/** - Workflows with human feedback
1. `01_basic_approval.py` - Basic approval workflow (15 min)
2. `02_timeout_checkpoint.py` - Timeout and checkpoint handling (20 min)
3. `03_channel_integration.py` - Feedback via channels (20 min)
4. `04_api_feedback.py` - REST API feedback submission (25 min)

### Level 13: Checkpoint/Resume 💾

**13_checkpoints/** - Workflow state persistence and fault tolerance
1. `01_basic_checkpoint.py` - Basic checkpoint/resume workflow (15 min)
2. `02_state_machine_checkpoint.py` - State machine with checkpoints (25 min)
3. `03_periodic_checkpoint.py` - Periodic checkpoints for long tasks (20 min)
4. `04_fault_recovery.py` - Fault tolerance with automatic retry (25 min)

**Total Learning Time**: ~14.5 hours to complete all examples

### Quick Start Path (30 minutes)

For a quick overview, follow this fast-track:
1. `01_basics/hello_world.py`
2. `02_workflows/simple_pipeline.py` ⭐
3. `03_data_flow/channels_basic.py`
4. `04_execution/direct_handler.py`

## Prerequisites

- Python 3.11 or higher
- Graflow installed (see Setup below)

### Setup

**Option 1: Using uv (recommended)**
```bash
# Install Graflow with development dependencies
uv sync --dev

# Install example-specific dependencies (e.g., for LLM integration)
cd examples/12_llm_integration
uv pip install -r requirements.txt
```

**Option 2: Install with all extras**
```bash
# Install Graflow with all optional dependencies
uv sync --dev --all-extras
```

### Optional Dependencies

Some examples require additional packages:

- **Redis examples**: `redis`
- **Docker examples**: `docker`
- **Visualization**: `grandalf`, `pygraphviz`, `requests`
- **LLM integration**: `litellm` (for LLMClient), `google-adk` (for agents)

## Running Examples

**Option 1: Direct execution**
```bash
uv run python examples/01_basics/hello_world.py
```

**Option 2: Using uv run (with dependencies)**
```bash
# Run with inline dependencies
uv run --with litellm python examples/12_llm_integration/simple_llm_client.py

# Or install from requirements.txt
cd examples/12_llm_integration
uv run --with-requirements requirements.txt python simple_llm_client.py
```

Expected output is documented in each example file's docstring.

## Project Structure

```
examples/
├── 01_basics/           # Fundamental concepts
│   ├── README.md       # Category documentation
│   ├── hello_world.py  # Simplest example
│   ├── task_dependencies.py
│   └── task_with_parameters.py
│
├── 02_workflows/        # Workflow orchestration
│   ├── README.md       # Category documentation
│   ├── simple_pipeline.py
│   ├── workflow_decorator.py
│   ├── operators_demo.py
│   ├── context_injection.py
│   └── task_graph_lowlevel_api.py
│
├── 03_data_flow/        # Inter-task communication
│   ├── README.md       # Category documentation
│   ├── channels_basic.py
│   ├── typed_channels.py
│   └── results_storage.py
│
├── 04_execution/        # Custom execution handlers
│   ├── README.md       # Category documentation
│   ├── direct_handler.py
│   ├── docker_handler.py
│   └── custom_handler.py
│
├── 05_distributed/      # Distributed execution
│   ├── README.md       # Category documentation
│   ├── redis_basics.py
│   ├── redis_worker.py
│   └── distributed_workflow.py
│
├── 06_advanced/         # Advanced patterns
│   ├── README.md       # Category documentation
│   ├── lambda_tasks.py
│   ├── custom_serialization.py
│   ├── nested_workflow.py
│   ├── global_context.py
│   ├── modular_etl.py
│   └── modular_etl/    # Task organization package
│       ├── __init__.py
│       ├── extract_tasks.py
│       ├── transform_tasks.py
│       └── load_tasks.py
│
├── 07_dynamic_tasks/    # Dynamic task generation
│   ├── README.md       # Category documentation
│   ├── dynamic_tasks.py
│   └── runtime_dynamic_tasks.py
│
├── 08_real_world/       # Real-world use cases
│   ├── README.md       # Category documentation
│   ├── data_pipeline.py
│   ├── ml_training.py
│   ├── batch_processing.py
│   └── sales_analysis.py
│
├── 09_visualization/    # Workflow visualization
│   ├── README.md       # Category documentation
│   ├── workflow_visualization.py
│   └── graph_utilities.py
│
├── 10_group_exec_policy/ # Parallel group error handling
│   ├── README.md       # Category documentation
│   ├── parallel_group_strict_mode.py
│   ├── parallel_group_best_effort.py
│   ├── parallel_group_at_least_n.py
│   ├── parallel_group_critical_tasks.py
│   └── parallel_group_custom_policy.py
│
├── 11_llm_integration/  # LLM integration
│   ├── README.md       # Category documentation
│   ├── simple_llm_client.py
│   ├── model_override.py
│   ├── llm_agent.py
│   └── multi_agent_workflow.py
│
├── 12_hitl/            # Human-in-the-Loop
│   ├── README.md       # Category documentation
│   ├── 01_basic_approval.py
│   ├── 02_timeout_checkpoint.py
│   ├── 03_channel_integration.py
│   └── 04_api_feedback.py
│
├── 13_checkpoints/     # Checkpoint/Resume
│   ├── README.md       # Category documentation
│   ├── 01_basic_checkpoint.py
│   ├── 02_state_machine_checkpoint.py
│   ├── 03_periodic_checkpoint.py
│   └── 04_fault_recovery.py
│
└── README.md           # This file
```

## Development Status

✅ Fully functional and tested with comprehensive documentation

### Progress Overview

| Phase | Status | Examples | Description |
|-------|--------|----------|-------------|
| 01_basics | ✅ Complete | 3/3 | Fundamental task concepts |
| 02_workflows | ✅ Complete | 5/5 | Workflow orchestration |
| 03_data_flow | ✅ Complete | 3/3 | Inter-task communication |
| 04_execution | ✅ Complete | 3/3 | Custom execution handlers |
| 05_distributed | ✅ Complete | 3/3 | Redis-based distribution |
| 06_advanced | ✅ Complete | 5/5 | Advanced patterns |
| 07_dynamic_tasks | ✅ Complete | 2/2 | Dynamic task generation |
| 08_real_world | ✅ Complete | 4/4 | Production use cases |
| 09_visualization | ✅ Complete | 2/2 | Workflow visualization |
| 10_group_exec_policy | ✅ Complete | 5/5 | Parallel group error handling policies |
| 11_llm_integration | ✅ Complete | 4/4 | LLM-powered workflows |
| 12_hitl | ✅ Complete | 4/4 | Human-in-the-Loop workflows |
| 13_checkpoints | ✅ Complete | 4/4 | Checkpoint/Resume for fault tolerance |

## Troubleshooting

### Import Errors

```python
ModuleNotFoundError: No module named 'graflow'
```

**Solution**: Install Graflow in development mode:
```bash
cd /path/to/graflow
uv sync --dev
```

### Redis Connection Errors

```python
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Solution**: Start Redis server:
```bash
# Using Docker
docker run -p 6379:6379 redis

# Using Homebrew (macOS)
brew services start redis

# Using apt (Ubuntu/Debian)
sudo service redis-server start
```

### Docker Permission Errors

```python
docker.errors.DockerException: Error while fetching server API version
```

**Solution**: Ensure Docker daemon is running and you have permissions:
```bash
# Start Docker Desktop (macOS/Windows)
# OR
sudo systemctl start docker  # Linux
```

## Contributing Examples

We welcome contributions! To add a new example:

1. Choose the appropriate category (or create a new one)
2. Follow the existing naming convention
3. Include comprehensive docstrings
4. Add expected output in comments
5. Test your example thoroughly
6. Update the category README.md

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.

## What's Next?

All planned examples are now complete! 🎉

The **47 examples** provide comprehensive coverage from basic concepts to AI-powered production applications. You can now:

1. **Build Production Workflows** - Use patterns from 08_real_world
2. **Scale with Redis** - Deploy distributed workflows from 05_distributed
3. **Apply Advanced Patterns** - Leverage techniques from 06_advanced
4. **Implement Runtime Dynamics** - Use next_task() and next_iteration() from 07_dynamic_tasks
5. **Analyze Data** - Build data analysis pipelines with anomaly detection
6. **Visualize Workflows** - Document and debug with ASCII, Mermaid, and PNG from 09_visualization
7. **Handle Parallel Errors** - Implement flexible error policies from 10_group_exec_policy
8. **Integrate LLMs** - Build AI-powered workflows with LLMClient and agents from 11_llm_integration
9. **Add Human Feedback** - Build interactive workflows with HITL from 12_hitl
10. **Implement Fault Tolerance** - Build resilient workflows with checkpoint/resume from 13_checkpoints

Additional examples may be added based on community feedback and emerging use cases.

## Tutorial Tests

All examples from the Tasks and Workflows Guide are also implemented as comprehensive unit tests in [`tests/tutorial/`](../tests/tutorial/):

- **63 tests** with **100% pass rate**
- **test_tasks_and_workflows_guide.py** - Core workflow features (34 tests)
- **test_llm_integration.py** - LLM client and agent injection (11 tests)
- **test_hitl.py** - Human-in-the-Loop feedback (18 tests)

These tests serve as both verification of functionality and practical code examples. They demonstrate proper usage patterns, mocking strategies, and best practices.

📋 **[View tutorial tests documentation →](../tests/tutorial/README.md)**

### Running Tutorial Tests

```bash
# Run all tutorial tests
uv run pytest tests/tutorial/ -v

# Run specific test file
uv run pytest tests/tutorial/test_tasks_and_workflows_guide.py -v
uv run pytest tests/tutorial/test_llm_integration.py -v
uv run pytest tests/tutorial/test_hitl.py -v
```

## API Notes

**Important**: The examples in this directory use stable, tested API patterns. All 47 examples are fully functional and production-ready. See [docs/examples_api_issues.md](../docs/examples_api_issues.md) for historical notes on API evolution.

## Getting Help

- 📖 [Main Documentation](../docs/)
- 🐛 [Report Issues](https://github.com/your-org/graflow/issues)
- 💬 [Discussions](https://github.com/your-org/graflow/discussions)

## License

These examples are part of the Graflow project and are licensed under the same terms.

---

**Happy Learning! 🚀**
