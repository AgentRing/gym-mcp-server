.PHONY: help install test lint format format-check typecheck check run-demo run-server run-agent-example

# Default target
help:
	@echo "Available targets:"
	@echo "  make install          - Install the package with uv"
	@echo "  make test             - Run all tests with coverage"
	@echo "  make lint             - Run linting with flake8"
	@echo "  make format           - Format code with black"
	@echo "  make format-check     - Check code formatting without modifying"
	@echo "  make typecheck        - Run type checking with mypy"
	@echo "  make check            - Run all checks (format, lint, typecheck, test)"
	@echo ""
	@echo "MCP Server (stdio transport):"
	@echo "  make run-server       - Run MCP server (stdio transport)"
	@echo "  make run-demo         - Run CartPole demo via stdio transport (random actions)"
	@echo "  make run-agent-example - Run OpenAI Agents SDK example (requires OPENAI_API_KEY)"

# Installation targets
install:
	@echo "📦 Installing gym-mcp-server..."
	uv pip install -e .

# Testing targets
test:
	@echo "🧪 Running tests with coverage..."
	uv run pytest

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

# MCP stdio server and example targets
run-server:
	@echo "🚀 Starting standalone MCP server with CartPole-v1..."
	@echo "Note: This runs a standalone server on stdio (for manual testing)."
	@echo "It waits for input on stdin. Most clients spawn their own server instead."
	@echo "For a working demo, use 'make run-demo' which spawns its own server."
	@echo ""
	uv run python -m gym_mcp_server --env CartPole-v1 --render-mode rgb_array --transport stdio

run-demo:
	@echo "🎮 Running CartPole demo via HTTP transport..."
	@echo "Note: The client spawns its own server in a background thread. All logs appear below."
	@echo ""
	uv run python examples/http_example.py

run-agent-example:
	@echo "🤖 Running OpenAI Agents SDK example..."
	@echo "Note: Requires OPENAI_API_KEY environment variable to be set."
	@echo "The AI agent will autonomously control the gym environment."
	@echo ""
	uv run python examples/openai_agents_stdio_example.py
