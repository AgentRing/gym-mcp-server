.PHONY: help install install-dev test test-unit test-integration test-verbose test-coverage lint format typecheck check run-demo run-cartpole run-mcp-client clean clean-all

# Default target
help:
	@echo "Available targets:"
	@echo "  make install          - Install the package with uv"
	@echo "  make install-dev      - Install with development dependencies"
	@echo "  make test             - Run all tests with coverage"
	@echo "  make test-unit        - Run only unit tests (skip integration)"
	@echo "  make test-integration - Run only integration tests"
	@echo "  make test-verbose     - Run tests with verbose output"
	@echo "  make test-coverage    - Run tests and open coverage report"
	@echo "  make lint             - Run linting with flake8"
	@echo "  make format           - Format code with black"
	@echo "  make format-check     - Check code formatting without modifying"
	@echo "  make typecheck        - Run type checking with mypy"
	@echo "  make check            - Run all checks (format, lint, typecheck, test)"
	@echo "  make run-demo         - Run CartPole in interactive mode"
	@echo "  make run-cartpole     - Run CartPole example"
	@echo "  make run-mcp-client   - Run MCP client example"
	@echo "  make clean            - Remove build artifacts and cache"
	@echo "  make clean-all        - Remove all generated files including htmlcov"

# Installation targets
install:
	@echo "📦 Installing gym-mcp-server..."
	uv pip install -e .

install-dev:
	@echo "📦 Installing gym-mcp-server with development dependencies..."
	uv pip install -e ".[dev,examples]"

# Testing targets
test:
	@echo "🧪 Running tests with coverage..."
	uv run pytest

test-unit:
	@echo "🧪 Running unit tests only..."
	uv run pytest -m "not integration"

test-integration:
	@echo "🧪 Running integration tests only..."
	uv run pytest -m integration

test-verbose:
	@echo "🧪 Running tests with verbose output..."
	uv run pytest -v

test-coverage:
	@echo "🧪 Running tests and generating coverage report..."
	uv run pytest
	@echo "📊 Opening coverage report..."
	xdg-open htmlcov/index.html 2>/dev/null || open htmlcov/index.html 2>/dev/null || echo "Coverage report generated at htmlcov/index.html"

# Code quality targets
lint:
	@echo "🔍 Running flake8..."
	uv run flake8 gym_mcp_server/ tests/ examples/

format:
	@echo "✨ Formatting code with black..."
	uv run black gym_mcp_server/ tests/ examples/

format-check:
	@echo "🔍 Checking code formatting..."
	uv run black --check gym_mcp_server/ tests/ examples/

typecheck:
	@echo "🔍 Running type checking with mypy..."
	uv run mypy gym_mcp_server/

check: format-check lint typecheck test
	@echo "✅ All checks passed!"

# Demo and example targets
run-demo:
	@echo "🎮 Running CartPole in interactive mode..."
	uv run python -m gym_mcp_server --env CartPole-v1 --interactive

run-cartpole:
	@echo "🎮 Running CartPole example..."
	uv run python examples/run_cartpole.py

run-mcp-client:
	@echo "🎮 Running MCP client example..."
	uv run python examples/mcp_client_example.py

# Cleanup targets
clean:
	@echo "🧹 Cleaning build artifacts and cache..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-all: clean
	@echo "🧹 Removing all generated files..."
	rm -rf htmlcov/
	rm -rf .coverage

