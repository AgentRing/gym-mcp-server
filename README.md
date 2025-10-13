# Gym MCP Server

Expose any Gymnasium environment as an MCP (Model Context Protocol) server, automatically converting the Gym API (`reset`, `step`, `render`) into MCP tools that any agent can call via standard JSON interfaces.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)](htmlcov/index.html)

## Features

- 🎮 Works with any Gymnasium environment
- 🔧 Exposes gym operations as MCP tools (`reset`, `step`, `render`, etc.)
- 🚀 Simple API with automatic serialization and error handling
- 🤖 Designed for AI agent integration (OpenAI Agents SDK, LangChain, etc.)
- 🔍 Type safe with full type hints

## Installation

```bash
pip install gym-mcp-server
```

**Requirements:** Python 3.12+

## Quick Start

### MCP Server 

Run the server with the standard MCP protocol:

```bash
# Using stdio transport (default)
python -m gym_mcp_server --env CartPole-v1 --transport stdio

# Using HTTP transport
python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --host localhost --port 8000

# Using SSE transport
python -m gym_mcp_server --env CartPole-v1 --transport sse --host localhost --port 8000
```

### Programmatic Usage

```python
from gym_mcp_server import GymMCPServer

# Create an MCP server with stdio transport
server = GymMCPServer(
    env_id="CartPole-v1",
    render_mode="rgb_array"
)

# Run the server (blocking call)
# server.run(transport="stdio")

# Or with HTTP transport
server_http = GymMCPServer(
    env_id="CartPole-v1",
    host="localhost",
    port=8000
)
# server_http.run(transport="streamable-http")
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
- **OpenAI Agents SDK (stdio)** - AI agent with stdio transport
- **OpenAI Agents SDK (HTTP)** - AI agent with HTTP transport

See [examples/README.md](examples/README.md) for details and instructions.

## Integration

### OpenAI Agents SDK

Use the `MCPServerStdio` or `MCPServerStreamableHttp` classes to connect agents to gym environments:

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

async with MCPServerStdio(
    name="Gym Environment",
    params={"command": "python", "args": ["-m", "gym_mcp_server", "--env", "CartPole-v1"]},
) as server:
    agent = Agent(name="GymAgent", instructions="...", mcp_servers=[server])
    result = await Runner.run(agent, "Play CartPole")
```

See [examples/openai_agents_stdio_example.py](examples/openai_agents_stdio_example.py) and [examples/openai_agents_http_example.py](examples/openai_agents_http_example.py).

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
- `--transport`: Transport type - stdio, streamable-http, or sse (default: stdio)
- `--host`: Host for HTTP-based transports (default: localhost)
- `--port`: Port for HTTP-based transports (default: 8000)

### Transport Options

The server supports multiple transport mechanisms:

**stdio** (Default): Standard input/output, suitable for local MCP clients
```bash
python -m gym_mcp_server --env CartPole-v1 --transport stdio
```

**streamable-http**: HTTP-based transport with streaming support
```bash
python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --host 0.0.0.0 --port 8000
```

**sse**: Server-Sent Events transport for real-time updates
```bash
python -m gym_mcp_server --env CartPole-v1 --transport sse --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Environment-Specific Dependencies

Some environments require additional packages:

```bash
pip install gymnasium[atari]   # For Atari environments
pip install gymnasium[box2d]   # For Box2D environments
pip install gymnasium[mujoco]  # For MuJoCo environments
```

### Python Version

Ensure you're using Python 3.12+:

```bash
python --version  # Should show 3.12 or higher
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

