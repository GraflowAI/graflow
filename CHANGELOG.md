# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.9] - 2026-04-16

Patch release to exclude vulnerable litellm versions (supply chain attack mitigation).

### Added
- Agent loop example (`examples/02_workflows/agent_loop.py`) demonstrating cyclic `agent >> tool >> agent` workflows with `terminate_workflow()`
- Multi-image Docker pipeline example (`examples/06_advanced/multi_image_docker.py`) showing per-task handler routing across different container images
- Tests for agent-loop pattern termination conditions

### Changed
- Exclude vulnerable `litellm` versions 1.82.7 and 1.82.8 to avoid supply chain attacks (#47)
- Bump `google-adk` minimum to v1.28.0

### Fixed
- Add `exclude-newer` and `exclude-newer-package` uv configuration for dependency cooldowns

## [0.1.8] - 2026-04-13

Patch release with Pydantic AI agent improvements and documentation updates.

### Changed
- Rename `create_pydantic_ai_agent_with_litellm` to `create_pydantic_ai_agent` for consistency (backward compatibility alias retained)
- Add `create_pydantic_ai_agent_with_litellm` to `__all__` for backward compatibility
- Add Ollama model support in `create_pydantic_ai_agent`

## [0.1.7] - 2026-04-07

Patch release with an important checkpoint bugfix and logging refinement.

### Fixed
- Add `_run_sync` helper to support nested event loops in Jupyter/Colab environments using asyncio
- Ensure deferred checkpoint includes successor tasks in execution queue (#46)

### Changed
- Change warning logs to debug logs for ADK and Pydantic agent import errors

## [0.1.6] - 2026-04-06

Minor patch release with a corner-case bugfix only. No new features or breaking changes.

### Fixed
- Fix result retrieval logic in `TaskWrapper` to prioritize `__result__` key (set by `context.set_result()`) over plain key when resolving task parameters

## [0.1.5] - 2026-04-06

### Added
- `atomic_add()` method for atomic arithmetic operations in `Channel` interface (#45)
- Advisory locking (`lock()`) in `Channel` interface for coordination between tasks (#45)
- Distributed advisory lock support in `RedisChannel` (#45)
- Thread-safe `append()` and `prepend()` methods in `MemoryChannel`
- Channel concurrency example in `examples/03_data_flow/channel_concurrency.py`
- Thread safety tests for `MemoryChannel`

## [0.1.4] - 2026-04-04

### Added
- `RetryPolicy` dataclass with exponential backoff support (`initial_interval`, `backoff_factor`, `max_interval`, `jitter`)
- `@task(retry_policy=RetryPolicy(...))` decorator parameter for per-task retry policy configuration
- `RetryController.set_node_policy()` and `get_policy_for_node()` methods
- Exponential backoff example in `examples/07_dynamic_tasks/task_retries.py`
- `RetryPolicy` exported from top-level `graflow` package

### Changed
- Iteration tasks now inherit `retry_policy` from base task (in addition to `max_retries`)
- `retry_policy` takes precedence over `max_retries` when both are set; `max_retries` is derived from the policy

## [0.1.3] - 2026-04-04

### Added
- `RetryController` for automatic task retry on failure (`graflow/core/retry.py`)
- `@task(max_retries=N)` decorator parameter for per-task retry configuration
- `ctx.retry_count` and `ctx.max_retries` properties on `TaskExecutionContext`
- `RetryLimitExceededError` raised when retry budget is exhausted
- Retry integration in `WorkflowEngine` with `_handle_task_retry()` method
- Iteration tasks inherit `max_retries` from base task
- `CycleController.accept_next_cycle()` for cycle budget checking
- `@task(max_cycles=N)` decorator parameter for per-task cycle limits
- 1-based `ctx.cycle_count` (first execution = 1)
- `task_iterations.py` and `task_retries.py` examples in `examples/07_dynamic_tasks/`
- Contributing guidelines (English and Japanese)

### Changed
- `CycleController` is now the single source of truth for cycle state
- `TaskExecutionContext` delegates cycle operations to `CycleController` via properties
- `default_max_retries` changed from 3 to 0 (no retries by default)
- Removed dead `retries` and `max_retries` instance variables from `TaskExecutionContext`
- Removed `register_cycle()` method from `TaskExecutionContext`
- Cycle count not incremented on retry (prevents double-counting)
- Rearranged imports and defined `__all__` for better module structure

### Fixed
- Version retrieval updated to use `importlib.metadata` for better compatibility

## [0.1.2] - 2026-02-08

### Fixed
- Simplify graph transformation by removing unnecessary checks for `ParallelGroup`
- Enhance parallel group task collection by including direct member tasks in graph transformations

## [0.1.1] - 2026-02-08

### Added
- Company Intelligence MCP Server with report generation and news search
- Webhook notification support for feedback events in HITL
- Open In Colab badge in README
- Initial README with project description and installation instructions

### Fixed
- Improve task existence check in `TaskGraph` to include `ParallelGroup`s
- Enhance feedback response handling to reject late responses
- Refactor successor processing logic in `WorkflowEngine` for clarity
- Allow feedback responses for requests with "timeout" status

## [0.1.0] - 2026-01-21

### Added
- Core task graph execution engine (`WorkflowEngine`)
- `@task` decorator for function-to-task conversion
- Workflow context with `>>` (sequential) and `|` (parallel) operators
- `ParallelGroup` for concurrent task execution
- `ExecutionContext` and `TaskExecutionContext` for state management
- Channel system for inter-task communication (memory and Redis backends)
- Distributed execution via Redis-based task queue and workers
- `CycleController` for cycle detection and prevention
- Dynamic task generation with `next_task()` and `next_iteration()`
- Checkpoint/resume system for long-running workflows
- Human-in-the-Loop (HITL) feedback system with multiple feedback types
- LLM integration with client and agent management
- Tracing support with Langfuse integration
- Task serialization via cloudpickle for distributed execution
- Workflow visualization (ASCII, Mermaid, PNG)
- REST API for workflow management and feedback submission

[Unreleased]: https://github.com/myui/graflow/compare/v0.1.9...HEAD
[0.1.9]: https://github.com/myui/graflow/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/myui/graflow/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/myui/graflow/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/myui/graflow/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/myui/graflow/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/myui/graflow/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/myui/graflow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/myui/graflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/myui/graflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/myui/graflow/releases/tag/v0.1.0
