# Graflow Examples

Welcome to the Graflow examples! This directory contains progressive examples to help you learn Graflow from basics to advanced use cases.

## 🎉 What's Available

**23 comprehensive, production-ready examples** covering:
- ✅ **Task Basics** - Define and execute tasks with parameters
- ✅ **Workflow Orchestration** - Sequential and parallel task composition
- ✅ **Data Flow** - Channels, typed communication, and result storage
- ✅ **Execution Control** - Direct, Docker, and custom handlers
- ✅ **Distributed Execution** - Redis-based task distribution across workers
- ✅ **Advanced Patterns** - Dynamic tasks, lambdas, and custom serialization
- ✅ **Real-World Use Cases** - Production-ready ETL, ML, and batch processing

All examples include detailed documentation, real-world use cases, and hands-on experiments!

## Quick Start

```bash
# Install Graflow
cd /path/to/graflow
pip install -e .

# Run your first example
python examples/01_basics/hello_world.py
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
- Dynamic task generation at runtime
- Lambda and closure tasks
- Nested workflow composition
- Custom serialization with cloudpickle

[View advanced examples →](06_advanced/)

### ✅ 07_real_world - Real-World Use Cases
**Status**: Complete | **Difficulty**: Intermediate to Advanced

Complete production-ready examples:
- ETL data pipeline with validation
- Machine learning training workflow
- Batch processing for large datasets

[View real-world examples →](07_real_world/)

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
2. `dynamic_tasks.py` - Runtime task generation (20 min)
3. `nested_workflow.py` - Hierarchical workflow organization (20 min)
4. `custom_serialization.py` - Understanding cloudpickle (15 min)

### Level 7: Production Use Cases 💼

**07_real_world/** - Complete real-world applications
1. `data_pipeline.py` - ETL workflow (20 min)
2. `ml_training.py` - ML training pipeline (20 min)
3. `batch_processing.py` - Large-scale batch processing (15 min)

**Total Learning Time**: ~6.1 hours to complete all examples

### Quick Start Path (30 minutes)

For a quick overview, follow this fast-track:
1. `01_basics/hello_world.py`
2. `02_workflows/simple_pipeline.py` ⭐
3. `03_data_flow/channels_basic.py`
4. `04_execution/direct_handler.py`

## Prerequisites

- Python 3.11 or higher
- Graflow installed (`pip install -e .` from the project root)

### Optional Dependencies

Some examples require additional packages:

- **Redis examples**: `pip install redis`
- **Docker examples**: `pip install docker`
- **Visualization**: `pip install matplotlib networkx`

## Running Examples

Each example is self-contained and can be run directly:

```bash
python examples/01_basics/hello_world.py
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
│   └── context_injection.py
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
│   ├── dynamic_tasks.py
│   └── custom_serialization.py
│
├── 07_real_world/       # Real-world use cases
│   ├── README.md       # Category documentation
│   ├── data_pipeline.py
│   ├── ml_training.py
│   └── batch_processing.py
│
└── README.md           # This file
```

## Development Status

✅ **Completed** (23 examples): Fully functional and tested with comprehensive documentation

### Progress Overview

| Phase | Status | Examples | Description |
|-------|--------|----------|-------------|
| 01_basics | ✅ Complete | 3/3 | Fundamental task concepts |
| 02_workflows | ✅ Complete | 4/4 | Workflow orchestration |
| 03_data_flow | ✅ Complete | 3/3 | Inter-task communication |
| 04_execution | ✅ Complete | 3/3 | Custom execution handlers |
| 05_distributed | ✅ Complete | 3/3 | Redis-based distribution |
| 06_advanced | ✅ Complete | 4/4 | Advanced patterns |
| 07_real_world | ✅ Complete | 3/3 | Production use cases |

**Total Progress**: 23/23 examples (100% complete) 🎉

## Troubleshooting

### Import Errors

```python
ModuleNotFoundError: No module named 'graflow'
```

**Solution**: Install Graflow in development mode:
```bash
cd /path/to/graflow
pip install -e .
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

The **23 examples** provide comprehensive coverage from basic concepts to production-ready applications. You can now:

1. **Build Production Workflows** - Use patterns from 07_real_world
2. **Scale with Redis** - Deploy distributed workflows from 05_distributed
3. **Apply Advanced Patterns** - Leverage techniques from 06_advanced

Additional examples may be added based on community feedback and emerging use cases.

## API Notes

**Important**: The examples in this directory use stable, tested API patterns. All 23 examples are fully functional and production-ready. See [docs/examples_api_issues.md](../docs/examples_api_issues.md) for historical notes on API evolution.

## Getting Help

- 📖 [Main Documentation](../docs/)
- 🐛 [Report Issues](https://github.com/your-org/graflow/issues)
- 💬 [Discussions](https://github.com/your-org/graflow/discussions)

## License

These examples are part of the Graflow project and are licensed under the same terms.

---

**Happy Learning! 🚀**
