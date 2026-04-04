# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/myui/graflow/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/myui/graflow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/myui/graflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/myui/graflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/myui/graflow/releases/tag/v0.1.0
