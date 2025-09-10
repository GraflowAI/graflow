# Repository Guidelines

## Project Structure & Module Organization
- `graflow/`: Source code.
  - `core/`: Workflow engine, tasks, graph, decorators.
  - `channels/`: In‑memory/Redis channels and typing helpers.
  - `queue/`: In‑memory/Redis task queues.
  - `coordination/`: Coordinators, executors, multiprocessing/Redis integration.
  - `worker/`: Worker entrypoints and handlers.
  - `utils/`, `exceptions.py`, `__init__.py`.
- `tests/`: Pytest suite by domain (e.g., `core/`, `queue/`, `scenario/`, `integration/`).
- `examples/`: Runnable examples (e.g., `examples/simple_workflow_example.py`).
- `docs/`: Architecture notes and dev plans.

## Build, Test, and Development Commands
- `make install` / `make install-dev`: Sync deps with `uv` (dev adds lint/test tools).
- `make format`: Auto‑fix common style issues via `ruff`.
- `make lint`: Run `ruff` and `mypy`.
- `make check`: Type check only (`mypy`).
- `make test`: Run unit tests (`pytest -v`).
- `make test-cov`: Run tests with coverage report.
- `make check-all`: Format, lint, and test.
- `make py examples/simple_test.py`: Run a file with `PYTHONPATH=.` using `uv`.

Prereqs: Python 3.11+, `uv` installed. Examples using Redis assume a local Redis at `localhost:6379`.

## Coding Style & Naming Conventions
- Indentation: 4 spaces; max line length: 120.
- Types: Add explicit annotations (mypy is strict on untyped defs).
- Naming: `snake_case` for functions/vars, `PascalCase` for classes, module names lowercase.
- Imports: Group and sort (`ruff`/isort settings; first‑party is `graflow`).

## Testing Guidelines
- Framework: `pytest`. Tests live under `tests/` and follow `test_*.py` naming.
- Markers: `@pytest.mark.integration` for external‑service tests (e.g., Redis). Example: `uvx pytest -m 'not integration'`.
- Run: `make test` locally before PRs. Use `make test-cov` for coverage when changing core logic.

## Commit & Pull Request Guidelines
- Commits: Use conventional prefixes seen in history: `feat:`, `fix:`, `refactor:`, `docs:`.
- Scope: Small, focused changes with clear messages.
- PRs must include: summary, rationale, linked issues, and test coverage for new/changed behavior. Update `examples/` and `docs/` when behavior changes.
- Pre‑submit: `make check-all` must pass; include screenshots or logs for UX/CLI changes where helpful.

## Security & Configuration Tips
- Redis examples: Prefer a sandboxed local instance. Do not commit credentials. For non‑default hosts/ports, pass parameters to `RedisChannel`/`RedisTaskQueue` constructors.
