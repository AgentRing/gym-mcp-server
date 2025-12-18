# Gym MCP Server

Expose any Gymnasium environment as an MCP (Model Context Protocol) server, automatically converting the Gym API (`reset`, `step`, `render`) into MCP tools that any agent can call via standard JSON interfaces.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)](htmlcov/index.html)

## Features

- 🎮 Works with any Gymnasium environment
- 🔧 Exposes gym operations via multiple protocols:
  - **MCP** (Model Context Protocol) over HTTP (`/mcp`, streamable-http)
  - **HTTP/REST** - FastAPI with Swagger UI (same server)
- 🚀 Simple API with automatic serialization and error handling
- 🤖 Designed for AI agent integration (OpenAI Agents SDK, LangChain, etc.)
- 🔍 Type safe with full type hints
- ♻️ Shared service layer for code reuse across protocols

## Installation

```bash
pip install gym-mcp-server
```

**Requirements:** Python 3.10+

## Quick Start

### Combined HTTP server (REST + MCP)

Run a single server that exposes both REST endpoints and the MCP endpoint:

```bash
python -m gym_mcp_server --env CartPole-v1 --host localhost --port 8000
# REST docs: http://localhost:8000/docs
# MCP endpoint: http://localhost:8000/mcp
```

### Programmatic Usage

```python
from gym_mcp_server import GymHTTPServer

# One HTTP server exposing both REST + MCP (/mcp) for the same env instance
server = GymHTTPServer(env_id="CartPole-v1", render_mode="rgb_array")
# server.run(host="localhost", port=8000)
```

## Available Tools

The server exposes these MCP tools:

- **`reset_env`** - Reset to initial state (optional `seed`)
- **`step_env`** - Take an action (required `action`)
- **`render_env`** - Render current state (optional `mode`)
- **`close_env`** - Close environment and free resources
- **`get_env_info`** - Get environment metadata
- **`get_available_tools`** - List all available tools

All tools return a standardized format:

```python
{
    "success": bool,  # Whether the operation succeeded
    "error": str,     # Error message (if success=False)
    # ... tool-specific data
}
```

## Examples

The [examples/](examples/) directory contains complete working examples:

- **MCP Server** - Creating and running MCP servers
- **MCP Client** - Low-level MCP protocol usage
- **HTTP Client** - Connecting to HTTP REST API
- **OpenAI Agents SDK (HTTP)** - AI agent with HTTP transport

See [examples/README.md](examples/README.md) for details and instructions.

## Integration

### OpenAI Agents SDK

Use the `MCPServerStreamableHttp` class to connect agents to gym environments:

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp

async with MCPServerStreamableHttp(
    name="Gym Environment",
    params={"url": "http://localhost:8000/mcp", "timeout": 10},
) as server:
    agent = Agent(name="GymAgent", instructions="...", mcp_servers=[server])
    result = await Runner.run(agent, "Play CartPole")
```

See [examples/openai_agents_http_example.py](examples/openai_agents_http_example.py).

Documentation: [OpenAI Agents SDK MCP Integration](https://openai.github.io/openai-agents-python/mcp/)

### Other Frameworks

Compatible with any MCP-compatible framework (LangChain, AutoGPT, custom MCP clients, etc.)

## Configuration

### Command Line Options

```bash
python -m gym_mcp_server --help
```

- `--env`: Gymnasium environment ID (required)
- `--render-mode`: Default render mode (e.g., rgb_array, human)
- `--host`: Host to bind (default: localhost)
- `--port`: Port to bind (default: 8000)

## Troubleshooting

### Environment-Specific Dependencies

Some environments require additional packages:

```bash
pip install gymnasium[atari]   # For Atari environments
pip install gymnasium[box2d]   # For Box2D environments
pip install gymnasium[mujoco]  # For MuJoCo environments
```

### Python Version

Ensure you're using Python 3.10+:

```bash
python --version  # Should show 3.10 or higher
```

## Development

For development and testing:

```bash
git clone https://github.com/haggaishachar/gym-mcp-server.git
cd gym-mcp-server
make install     # Install with dependencies
make check       # Run all checks (format, lint, typecheck, test)
```

See the [Makefile](Makefile) for all available commands.

## License

MIT License - see the LICENSE file for details.

## Links

- [GitHub Repository](https://github.com/haggaishachar/gym-mcp-server)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

