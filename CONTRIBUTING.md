# Contributing to Graflow

Thank you for your interest in contributing to Graflow! Your involvement helps make Graflow better for everyone. This guide covers all the ways you can contribute.

## Open Meritocracy

Graflow follows the principles of [Apache-style meritocracy](https://www.apache.org/foundation/how-it-works/#meritocracy) — roles and responsibilities are earned through demonstrated contributions, not assigned by title or affiliation.

We actively welcome contributors to grow into project members:

| Role | Description |
|------|-------------|
| **Contributor** | Anyone who submits patches, bug reports, documentation, or participates in discussions. |
| **Committer** | Contributors who have earned write access through sustained, high-quality contributions. |
| **Maintainer** | Committers who take on project-wide stewardship: reviewing PRs, guiding roadmap, and mentoring newcomers. |

There is no fixed process to "apply" for these roles — they are recognized by your peers based on merit. If you contribute consistently and constructively, you will be invited to take on greater responsibility. Every contribution matters, whether it's a one-line typo fix or a major new feature.

## Ways to Contribute

- **Report bugs** - Found something broken? [Open an issue](#reporting-bugs).
- **Suggest features** - Have an idea? [Start a discussion](#suggesting-features).
- **Improve documentation** - Fix typos, add examples, clarify explanations.
- **Contribute code** - Bug fixes, new features, performance improvements.
- **Add integrations** - New queue/channel backends, LLM providers, tracing integrations.

Look for issues labeled [`good first issue`](https://github.com/GraflowAI/graflow/labels/good%20first%20issue) or [`help wanted`](https://github.com/GraflowAI/graflow/labels/help%20wanted) to find accessible entry points.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Setup

```bash
# Fork and clone the repository
git clone https://github.com/<your-username>/graflow.git
cd graflow

# Install development dependencies
uv sync --dev

# (Optional) Install all extras including graphviz (macOS)
make install-all
```

### Verify Your Setup

```bash
make check-all  # Runs format + lint + test
```

## Development Workflow

### 1. Find or Create an Issue

Before starting work, check [existing issues](https://github.com/GraflowAI/graflow/issues) to avoid duplicate effort. If you plan to work on something, comment on the issue to let others know.

### 2. Fork and Create a Branch

```bash
git checkout -b your-feature-name
```

Use a descriptive branch name (e.g., `fix/parallel-group-merge`, `feat/kafka-backend`).

### 3. Make Your Changes

Follow the project's [code style](#code-style). Keep your changes focused — a PR should address a single concern.

### 4. Run Quality Checks

All checks must pass before submitting a PR:

```bash
make format     # Auto-fix formatting with ruff
make lint       # Run ruff linter
make test       # Run pytest suite
```

Or run all checks at once:

```bash
make check-all
```

### 5. Commit

Write clear, concise commit messages that describe the "why" over the "what".

```bash
# Good
git commit -m "fix: Prevent duplicate task execution in ParallelGroup"

# Bad
git commit -m "fixed stuff"
```

### 6. Open a Pull Request

Open a PR against the `main` branch. In your PR description:

- Summarize what the change does and why
- Reference related issues (e.g., `Fixes #42`)
- Describe how you tested it

## Pull Request Guidelines

### General

- **Keep scope small**: Each PR should address a single concern. Avoid mixing bug fixes with refactors or new features.
- **Backwards compatibility**: Your changes must not break existing behavior, except for critical bug or security fixes.
- **No unnecessary dependencies**: Do not add new hard dependencies without discussion. Optional dependencies should go in `[project.optional-dependencies]`.
- **Tests required**: Bug fixes should include a test that reproduces the bug. New features must include unit tests and, if appropriate, integration tests.
- **Documentation**: New features should include docstrings and, if significant, updates to examples.

### What We Look For in Review

- Code is clear and follows project conventions
- Tests cover the important cases
- No unnecessary complexity or over-engineering
- Type hints are present on all public APIs
- Changes are backward-compatible

### Bug Fixes

Include a clear explanation of the bug and a test that reproduces it. The fix should be minimal and focused.

### New Features

New features have a higher bar for acceptance. Before spending significant time, open an issue to discuss the approach. New features must include:

- Unit tests (and integration tests where appropriate)
- Docstrings for public APIs
- An example in `examples/` if the feature introduces a new usage pattern

### AI-Assisted Contributions

We welcome the use of AI tools to assist with contributions, but AI assistance must be paired with meaningful human review and understanding. If the effort to create a PR is less than the effort to review it, that contribution should not be submitted. Low-effort, AI-generated PRs without human judgment may be closed without review.

## Reporting Bugs

Search [existing issues](https://github.com/GraflowAI/graflow/issues) first. If your bug hasn't been reported, open a new issue with:

- A clear, descriptive title
- A **minimal reproducible example**
- Steps to reproduce
- Expected vs. actual behavior
- Python version, OS, and Graflow version (`python -c "import graflow; print(graflow.__version__)"`)
- Relevant logs or error messages

Keep issues focused on a single bug. If you find multiple issues, open separate tickets.

## Suggesting Features

Search [existing issues](https://github.com/GraflowAI/graflow/issues) and [discussions](https://github.com/GraflowAI/graflow/discussions) first. When suggesting a feature:

- Describe the use case and motivation
- Provide concrete examples of how it would be used
- Explain why existing functionality doesn't meet the need

## Code Style

### Formatting & Linting

- **Formatter/Linter**: [ruff](https://docs.astral.sh/ruff/)
- **Type checker**: [mypy](https://mypy-lang.org/)
- **Line length**: 120 characters
- **Import sorting**: isort via ruff, with `graflow` as known first-party

### Type Hints

- Required for all function signatures (`disallow_untyped_defs = true`)
- No implicit optionals (`no_implicit_optional = true`)
- Avoid `Any` when possible

### Naming Conventions

| Element   | Style                 | Example                          |
|-----------|-----------------------|----------------------------------|
| Classes   | `PascalCase`          | `TaskQueue`, `ExecutionContext`   |
| Functions | `snake_case`          | `get_next_task`, `add_to_queue`  |
| Constants | `UPPER_SNAKE_CASE`    | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Private   | `_leading_underscore` | `_internal_method`               |

### Docstrings

Public classes and methods should have docstrings with `Args`, `Returns`, and `Raises` sections where applicable.

## Testing

Tests live in the `tests/` directory, organized by module:

```
tests/
    core/             # Core functionality
    channels/         # Channel implementations
    coordination/     # Coordination tests
    queue/            # Queue implementations
    worker/           # Worker tests
    integration/      # Integration tests
    scenario/         # End-to-end scenarios
    unit/             # Unit tests
```

### Running Tests

```bash
# All tests
uv run pytest tests/ -v

# Specific directory
uv run pytest tests/core/ -v

# Specific test file
uv run pytest tests/core/test_task.py -v

# Specific test
uv run pytest tests/core/test_task.py::test_name -v
```

### Writing Tests

- Use `pytest` fixtures (see `conftest.py` files)
- Mock external dependencies (Redis, Docker)
- Test both memory and Redis backends for queues/channels where relevant
- Add integration tests for distributed scenarios

## Project Structure

```
graflow/
    core/           # Engine, tasks, workflows, context, graph
    queue/          # Task queue backends (memory, Redis)
    channels/       # Inter-task communication (memory, Redis)
    worker/         # Distributed task workers
    coordination/   # Parallel execution coordination
    hitl/           # Human-in-the-loop feedback
    llm/            # LLM integration
    trace/          # Tracing and observability
    api/            # REST API endpoints
    serialization/  # Task serialization (cloudpickle)
    debug/          # Visualization and debugging
```

## Common Contribution Areas

Check the [project roadmap](https://github.com/orgs/GraflowAI/projects/1) for planned features and priorities. Items on the Kanban board are great candidates for contribution — pick one up and comment on the issue to get started.

### Adding Agent Framework Integrations

Graflow currently supports [ADK (Google Agent Development Kit)](https://google.github.io/adk-docs/) and [PydanticAI](https://ai.pydantic.dev/) in `graflow/llm/`. Adding support for other agent frameworks is highly encouraged. Examples include:

- [Strands Agents](https://strandsagents.com/latest/) (AWS)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- Any other agent or super-agent framework

To add a new integration, implement an adapter in `graflow/llm/` following the patterns established by the existing ADK and PydanticAI integrations.

### Improving Documentation

Documentation contributions are always welcome:

- Fix typos, clarify explanations, improve readability
- Add or improve docstrings in source code
- Add new examples to `examples/`
- Propose structural changes or new documentation pages

## Getting Help

- Open a [GitHub Discussion](https://github.com/GraflowAI/graflow/discussions) for questions
- Check existing issues and discussions before asking
- Tag maintainers if you're stuck on a PR

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](./LICENSE).
