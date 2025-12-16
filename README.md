# Gym MCP Server

Expose any Gymnasium environment as an MCP (Model Context Protocol) server, automatically converting the Gym API (`reset`, `step`, `render`) into MCP tools that any agent can call via standard JSON interfaces.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen.svg)](htmlcov/index.html)

## Features

- 🎮 Works with any Gymnasium environment
- 🔧 Exposes gym operations via multiple protocols:
  - **MCP** (Model Context Protocol) - stdio, HTTP, SSE transports
  - **HTTP/REST** - FastAPI with Swagger UI
  - **gRPC** - High-performance RPC with protobuf
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

### HTTP/REST Server

Run the server as a REST API with Swagger UI:

```bash
# Using HTTP REST API with Swagger UI
python -m gym_mcp_server --env CartPole-v1 --http --port 8000
# Then open http://localhost:8000/docs for Swagger UI
```

### gRPC Server

Run the server as a gRPC service:

```bash
# Using gRPC (default port 50051)
python -m gym_mcp_server --env CartPole-v1 --grpc --port 50051
```

### Programmatic Usage

```python
from gym_mcp_server import GymMCPServer, GymHTTPServer, GymGRPCServer

# MCP server with stdio transport
mcp_server = GymMCPServer(env_id="CartPole-v1", render_mode="rgb_array")
# mcp_server.run(transport="stdio")

# HTTP server with Swagger UI
http_server = GymHTTPServer(env_id="CartPole-v1")
# http_server.run(host="localhost", port=8000)

# gRPC server
grpc_server = GymGRPCServer(env_id="CartPole-v1")
# grpc_server.run(host="localhost", port=50051)
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
- **gRPC Client** - Connecting to gRPC server
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
- `--transport`: Transport type - stdio, streamable-http, or sse (default: stdio, MCP only)
- `--http`: Run as HTTP REST API server with Swagger UI
- `--grpc`: Run as gRPC server
- `--host`: Host for HTTP/gRPC servers (default: localhost)
- `--port`: Port for HTTP/gRPC servers (default: 8000 for HTTP, 50051 for gRPC)

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

**http**: HTTP REST API with Swagger UI
```bash
python -m gym_mcp_server --env CartPole-v1 --http --host 0.0.0.0 --port 8000
# Access Swagger UI at http://0.0.0.0:8000/docs
```

**grpc**: gRPC server with protobuf
```bash
python -m gym_mcp_server --env CartPole-v1 --grpc --host 0.0.0.0 --port 50051
```

### Generating gRPC Code

Before using the gRPC server, you need to generate Python code from the proto file:

```bash
# Install dev dependencies (includes grpcio-tools)
pip install -e ".[dev]"

# Generate proto code
python scripts/generate_proto.py
```

Or manually:
```bash
python -m grpc_tools.protoc -I gym_mcp_server/proto \
    --python_out=gym_mcp_server --grpc_python_out=gym_mcp_server \
    gym_mcp_server/proto/gym_service.proto
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

