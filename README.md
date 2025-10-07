# Gym MCP Server

Expose any Gymnasium environment as an MCP (Model Context Protocol) server, automatically converting the Gym API (`reset`, `step`, `render`) into MCP tools that any agent can call via standard JSON interfaces.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](htmlcov/index.html)

## 📖 Table of Contents

- [Features](#-features)
- [Requirements](#-requirements)
- [Quick Start in 60 Seconds](#-quick-start-in-60-seconds)
- [Design Overview](#️-design-overview)
- [Installation & Usage](#-quick-start)
  - [Installation](#installation)
  - [Basic Usage](#basic-usage)
- [Core Implementation](#-core-implementation)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Development](#️-development)
  - [Makefile Commands](#makefile-commands)
  - [Recommended Workflow](#recommended-workflow)
  - [Project Structure](#project-structure)
  - [Test Coverage](#test-coverage)
  - [Code Quality Standards](#code-quality-standards)
- [Contributing](#-contributing)
- [AI Agent Integration](#-integration-with-ai-agents)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

## ✨ Features

- 🎮 **Universal Gym Interface**: Works with any Gymnasium environment
- 🔧 **MCP Tools**: Exposes gym operations (`reset`, `step`, `render`, etc.) as MCP tools
- 🚀 **Easy to Use**: Simple API with automatic serialization and error handling
- 📦 **Modern Python**: Built with Python 3.12+ for optimal performance
- 🧪 **Well Tested**: 93% test coverage with comprehensive test suite
- 🛠️ **Developer Friendly**: Full Makefile with all common tasks
- 🔍 **Type Safe**: Full type hints with mypy checking

## 📋 Requirements

- **Python 3.12+** (required for modern features and performance)
- **uv** (recommended) or **pip** for package management
- **Gymnasium** 0.29.0+

## ⚡ Quick Start in 60 Seconds

```bash
# 1. Clone and setup
git clone <repository-url> && cd gym-mcp-server
make install-dev

# 2. Run interactive demo
make run-demo

# 3. Run all checks
make check
```

That's it! You now have a working MCP server for Gymnasium environments. 🎉

## ⚙️ Design Overview

### Architecture

```
[MCP Client / Agent SDK]
        ↓
  (calls MCP tools)
        ↓
[ gym-mcp-server ]
        ↓
[ any gym.Env subclass ]
```

You can use this with any Gym environment, e.g.:

```bash
python -m gym_mcp_server --env CartPole-v1
python -m gym_mcp_server --env WebShop-v0
```

## 📁 Directory Layout

```
gym-mcp-server/
│
├── gym_mcp_server/
│   ├── __init__.py
│   ├── server.py         # main MCP server exposing a generic Gym environment
│   ├── utils.py          # serialization, rendering, etc.
│   └── schemas.py        # JSON schema definitions for observations/actions
│
├── examples/
│   ├── run_cartpole.py   # minimal example using MCP client
│   └── mcp_client_example.py  # comprehensive MCP client example
│
├── pyproject.toml
└── README.md
```

## 🚀 Quick Start

### Installation

#### Prerequisites

Ensure you have **Python 3.12 or higher** installed:

```bash
python --version  # Should show 3.12 or higher
```

#### Option 1: Using uv (Recommended) 🚀

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd gym-mcp-server

# Install with dependencies (using Makefile)
make install

# Or install with development dependencies
make install-dev
```

#### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd gym-mcp-server

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,examples]"
```

### Basic Usage

#### 1. Interactive Mode

Run the server in interactive mode to test it:

```bash
# Using Makefile (recommended)
make run-demo

# Or directly with uv
uv run python -m gym_mcp_server --env CartPole-v1 --interactive

# Or with pip
python -m gym_mcp_server --env CartPole-v1 --interactive
```

This will start an interactive session where you can manually control the environment:

```
Running interactive Gym MCP adapter for CartPole-v1
Available commands:
  reset [seed] - Reset the environment
  step <action> - Take an action
  render [mode] - Render the environment
  info - Get environment information
  close - Close the environment
  quit - Exit the program

> reset
Reset result: {
  "observation": [0.012, -0.034, 0.045, 0.067],
  "info": {},
  "done": false,
  "success": true
}
> step 1
Step result: {
  "observation": [0.015, 0.123, 0.042, -0.234],
  "reward": 1.0,
  "done": false,
  "truncated": false,
  "info": {},
  "success": true
}
```

#### 2. Programmatic Usage

```python
from gym_mcp_server import GymMCPAdapter

# Initialize the adapter
adapter = GymMCPAdapter("CartPole-v1")

# Reset the environment
reset_result = adapter.call_tool("reset_env")
print(f"Initial observation: {reset_result['observation']}")

# Take an action
step_result = adapter.call_tool("step_env", action=1)
print(f"Reward: {step_result['reward']}")

# Render the environment
render_result = adapter.call_tool("render_env")
print(f"Render output: {render_result['render']}")

# Close the environment
adapter.call_tool("close_env")
```

#### 3. Run Examples

```bash
# Using Makefile (recommended)
make run-cartpole        # Run CartPole example
make run-mcp-client      # Run MCP client example

# Or directly with uv
uv run python examples/run_cartpole.py
uv run python examples/mcp_client_example.py

# Or with pip
python examples/run_cartpole.py
python examples/mcp_client_example.py
```

## 🧠 Core Implementation

### Available MCP Tools

The server exposes the following tools:

1. **`reset_env`** - Reset the environment to its initial state
   - Parameters: `seed` (optional integer)
   - Returns: Initial observation, info, and done status

2. **`step_env`** - Take an action in the environment
   - Parameters: `action` (action to take)
   - Returns: Next observation, reward, done status, and info

3. **`render_env`** - Render the current state of the environment
   - Parameters: `mode` (optional render mode)
   - Returns: Rendered output (text, image, or array)

4. **`close_env`** - Close the environment and free resources
   - Parameters: None
   - Returns: Close status

5. **`get_env_info`** - Get information about the environment
   - Parameters: None
   - Returns: Environment metadata (action space, observation space, etc.)

6. **`get_available_tools`** - Get information about available MCP tools
   - Parameters: None
   - Returns: Tool schemas and descriptions

### Serialization

The server automatically handles serialization of:

- **Observations**: Converted to JSON-safe formats (lists, dicts, primitives)
- **Actions**: Converted to appropriate types for the environment
- **Render Output**: Images are base64-encoded, arrays are converted to lists
- **Rewards**: Converted to float values
- **Info**: Dictionary containing additional environment information

### Error Handling

All tools return a standardized response format:

```python
{
    "success": bool,      # Whether the operation succeeded
    "error": str,         # Error message (if success=False)
    # ... tool-specific data
}
```

## 🔧 Configuration

### Environment Variables

- `GYM_MCP_HOST`: Host to bind to (default: localhost)
- `GYM_MCP_PORT`: Port to bind to (default: 8000)
- `GYM_MCP_RENDER_MODE`: Default render mode (default: ansi)

### Command Line Options

```bash
python -m gym_mcp_server --help
```

Options:
- `--env`: Gymnasium environment ID (required)
- `--render-mode`: Default render mode
- `--interactive`: Run in interactive mode
- `--host`: Host to bind to
- `--port`: Port to bind to

## 📚 Examples

### Example 1: Simple CartPole Control

```python
from gym_mcp_server import GymMCPAdapter

adapter = GymMCPAdapter("CartPole-v1")

# Reset environment
obs = adapter.call_tool("reset_env")["observation"]
done = False
step_count = 0

while not done and step_count < 100:
    # Simple policy: move right if pole angle is positive
    action = 1 if obs[2] > 0 else 0
    
    result = adapter.call_tool("step_env", action=action)
    obs = result["observation"]
    done = result["done"]
    step_count += 1
    
    print(f"Step {step_count}: Reward={result['reward']:.2f}")

print(f"Episode finished after {step_count} steps")
adapter.call_tool("close_env")
```

### Example 2: Random Agent

```python
import random
from gym_mcp_server import GymMCPAdapter

adapter = GymMCPAdapter("CartPole-v1")

# Run multiple episodes
for episode in range(5):
    obs = adapter.call_tool("reset_env")["observation"]
    done = False
    total_reward = 0
    
    while not done:
        action = random.randint(0, 1)  # Random action
        result = adapter.call_tool("step_env", action=action)
        obs = result["observation"]
        done = result["done"]
        total_reward += result["reward"]
    
    print(f"Episode {episode + 1}: Total reward = {total_reward:.2f}")

adapter.call_tool("close_env")
```

### Example 3: Environment Information

```python
from gym_mcp_server import GymMCPAdapter

adapter = GymMCPAdapter("CartPole-v1")

# Get environment information
info = adapter.call_tool("get_env_info")
print("Environment Information:")
print(f"  ID: {info['env_info']['id']}")
print(f"  Action Space: {info['env_info']['action_space']}")
print(f"  Observation Space: {info['env_info']['observation_space']}")
print(f"  Action Space Size: {info['env_info']['action_space_size']}")

# Get available tools
tools = adapter.call_tool("get_available_tools")
print("\nAvailable Tools:")
for tool_name, tool_info in tools["tools"].items():
    print(f"  {tool_name}: {tool_info['description']}")

adapter.call_tool("close_env")
```

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone <repository-url>
cd gym-mcp-server

# Install with development dependencies
make install-dev
```

### Makefile Commands

The project includes a comprehensive Makefile with all common development tasks:

#### Installation

```bash
make install          # Install the package
make install-dev      # Install with development dependencies
```

#### Testing

```bash
make test             # Run all tests with coverage
make test-unit        # Run only unit tests (skip integration)
make test-integration # Run only integration tests
make test-verbose     # Run tests with verbose output
make test-coverage    # Run tests and open coverage report
```

#### Code Quality

```bash
make format           # Format code with black
make format-check     # Check code formatting without modifying
make lint             # Run linting with flake8
make typecheck        # Run type checking with mypy
make check            # Run ALL checks (format, lint, typecheck, test)
```

#### Running Examples

```bash
make run-demo         # Run CartPole in interactive mode
make run-cartpole     # Run CartPole example
make run-mcp-client   # Run MCP client example
```

#### Cleanup

```bash
make clean            # Remove build artifacts and cache
make clean-all        # Remove all generated files including htmlcov
```

#### Quick Reference

```bash
make help             # Show all available targets
```

### Recommended Workflow

1. **Before making changes:**
   ```bash
   make check  # Ensure everything is working
   ```

2. **While developing:**
   ```bash
   make format      # Format your code
   make test-unit   # Run unit tests quickly
   ```

3. **Before committing:**
   ```bash
   make check  # Run all checks (format, lint, typecheck, test)
   ```

### Project Structure

```
gym-mcp-server/
├── gym_mcp_server/          # Main package
│   ├── __init__.py          # Package initialization
│   ├── server.py            # Core MCP server (95% coverage)
│   ├── utils.py             # Utilities (88% coverage)
│   └── schemas.py           # JSON schemas (100% coverage)
├── examples/                 # Example scripts
│   ├── run_cartpole.py      # CartPole example
│   └── mcp_client_example.py # MCP client example
├── tests/                    # Test suite (92 tests, 93% coverage)
│   ├── conftest.py          # Pytest configuration
│   ├── test_init.py         # Package tests
│   ├── test_schemas.py      # Schema tests
│   ├── test_server.py       # Server tests
│   ├── test_utils.py        # Utility tests
│   └── test_integration.py  # Integration tests
├── Makefile                  # Development commands
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

### Test Coverage

Current test coverage: **93%**

- `__init__.py`: 100%
- `schemas.py`: 100%
- `server.py`: 95%
- `utils.py`: 88%

### Code Quality Standards

- **Formatter**: Black (line length: 88)
- **Linter**: Flake8
- **Type Checker**: mypy with strict settings
- **Test Framework**: pytest with coverage reporting
- **Python Version**: 3.12+ (using modern features)

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the repository** and clone it locally

2. **Set up your development environment:**
   ```bash
   cd gym-mcp-server
   make install-dev
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes** and add tests for new functionality

5. **Run all checks before committing:**
   ```bash
   make check  # Runs format-check, lint, typecheck, and tests
   ```

6. **Fix any issues:**
   ```bash
   make format     # Format code if needed
   make test       # Ensure all tests pass
   ```

7. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

8. **Push and submit a pull request:**
   ```bash
   git push origin feature/your-feature-name
   ```

### Contribution Guidelines

- Follow the existing code style (enforced by Black and Flake8)
- Add type hints to all new functions
- Write tests for new functionality (maintain 90%+ coverage)
- Update documentation as needed
- Run `make check` before submitting PR
- Write clear commit messages

## 🤖 Integration with AI Agents

This project is designed to work seamlessly with AI agent frameworks like the [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/):

### OpenAI Agents SDK Compatibility

The project requires Python 3.12+, which is fully compatible with the OpenAI Agents SDK (requires Python 3.9+).

```bash
# Install both packages
pip install gym-mcp-server openai-agents

# Use with OpenAI Agents
from agents import Agent, Runner
from gym_mcp_server import GymMCPAdapter

# Create your agent-controlled gym environment
adapter = GymMCPAdapter("CartPole-v1")
agent = Agent(name="GymAgent", instructions="Control the CartPole environment")

# Your agent can now interact with the gym environment through MCP tools
```

### MCP Integration

The server exposes Gymnasium environments via the Model Context Protocol (MCP), making it easy to integrate with:

- **AI Agents**: OpenAI Agents SDK, LangChain, AutoGPT, etc.
- **LLM Applications**: Any application that supports MCP
- **Research Tools**: Custom RL research pipelines
- **Multi-Agent Systems**: Coordinate multiple agents in gym environments

## 🔧 Troubleshooting

### Python Version Issues

**Problem**: `ImportError` or version-related errors

**Solution**: Ensure you're using Python 3.12 or higher:
```bash
python --version  # Check version
pyenv install 3.12  # Install if needed (with pyenv)
```

### Installation Issues with uv

**Problem**: `uv` command not found

**Solution**: Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart your terminal
```

### Test Failures

**Problem**: Tests fail after making changes

**Solution**: Run checks individually to identify the issue:
```bash
make format-check  # Check formatting
make lint          # Check linting
make typecheck     # Check types
make test-verbose  # Run tests with details
```

### Coverage Issues

**Problem**: Coverage report not opening

**Solution**: Manually open the coverage report:
```bash
make test  # Generate coverage
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html       # macOS
```

### Environment-Specific Issues

**Problem**: Certain Gym environments don't load

**Solution**: Install environment-specific dependencies:
```bash
pip install gymnasium[atari]  # For Atari environments
pip install gymnasium[box2d]  # For Box2D environments
pip install gymnasium[mujoco] # For MuJoCo environments
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Gymnasium](https://gymnasium.farama.org/) for the excellent RL environment framework
- [Model Context Protocol](https://modelcontextprotocol.io/) for the MCP specification
- The open-source community for inspiration and contributions

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-username/gym-mcp-server/issues) page
2. Create a new issue with detailed information
3. Join our community discussions

---

**Happy Reinforcement Learning with MCP! 🚀**