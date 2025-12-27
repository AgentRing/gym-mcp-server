.PHONY: help install test lint format typecheck check run-server

# Default target
help:
	@echo "Available targets:"
	@echo "  make install          - Install the package with uv"
	@echo "  make test             - Run all tests with coverage"
	@echo "  make lint             - Run linting with flake8"
	@echo "  make format           - Format code with black"
	@echo "  make typecheck        - Run type checking with mypy"
	@echo "  make check            - Run all checks (lint, typecheck, test)"
	@echo ""
	@echo "Server:"
	@echo "  make run-server       - Run combined HTTP server (REST + MCP) for CartPole-v1"

# Installation targets
install:
	@echo "📦 Installing gym-mcp-server..."
	@if [ ! -d ".venv" ]; then uv venv; fi
	uv pip install -e ".[dev]"

# Testing targets
test:
	@echo "🧪 Running tests with coverage..."
	@echo "📦 Ensuring dev dependencies are installed..."
	@uv pip install -e ".[dev]" || true
	@uv run pytest

# Code quality targets
lint:
	@echo "🔍 Running flake8..."
	uv run flake8 gym_mcp_server/ tests/

format:
	@echo "✨ Formatting code with black..."
	uv run black gym_mcp_server/ tests/

typecheck:
	@echo "🔍 Running type checking with mypy..."
	uv run mypy gym_mcp_server/

check: lint typecheck test
	@echo "✅ All checks passed!"

# Server target
run-server:
	@echo "🚀 Starting combined HTTP server (REST + MCP) with CartPole-v1..."
	@echo "REST docs: http://localhost:8000/docs"
	@echo "MCP endpoint: http://localhost:8000/mcp"
	@echo ""
	uv run python -m gym_mcp_server --env CartPole-v1 --render-mode rgb_array --host localhost --port 8000
