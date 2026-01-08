.PHONY: format lint check test clean install py

# Format code using ruff
format:
	uvx ruff check --fix --select W293,W291,I001,F401,F541,W292,F541 --unsafe-fixes .
	uvx ruff format .

# Run linting
lint:
	uvx ruff check .
#	uvx mypy graflow/

# Type check only
check:
	uvx mypy .

# Run tests
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ --cov=flowlet --cov-report=html --cov-report=term

# Clean up cache and build artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/

# Install dependencies
install:
	uv sync

# Install development dependencies
install-dev:
	uv sync --dev

# Install recommended extras (e.g., graphviz)
install-all:
	export CFLAGS="-I$(brew --prefix graphviz)/include"
	export LDFLAGS="-L$(brew --prefix graphviz)/lib"
	uv sync --all-extras --dev

# Run all checks (format, lint, test)
check-all: format lint test

# Quick fix for common issues
fix:
	uvx ruff check --fix --unsafe-fixes .
	uvx ruff format .

# Run Python files with proper environment setup
# Usage: make py examples/simple_test.py
py:
	PYTHONPATH=. uv run python $(filter-out $@,$(MAKECMDGOALS))

# Catch-all target to prevent make from complaining about unknown targets
%:
	@:

# Help
help:
	@echo "Available targets:"
	@echo "  format         - Format code using ruff"
	@echo "  lint           - Run linting with ruff and mypy"
	@echo "  check          - Run type checking with mypy"
	@echo "  test           - Run tests"
	@echo "  test-cov       - Run tests with coverage"
	@echo "  clean          - Clean up cache and build artifacts"
	@echo "  install    	- Install dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  install-extras - Install extras (e.g., graphviz)"
	@echo "  check-all      - Run format, lint, and test"
	@echo "  fix            - Quick fix for common issues"
	@echo "  py <file>      - Run Python file with proper environment (e.g., make py examples/simple_test.py)"
	@echo "  help           - Show this help message"
