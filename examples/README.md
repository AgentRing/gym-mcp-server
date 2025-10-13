# Gym MCP Server Examples

This directory contains examples demonstrating different ways to use the gym-mcp-server.

## Examples

### 1. MCP Server Example

Create and run a FastMCP server using `GymMCPServer` class.

**What it demonstrates:**
- Using `GymMCPServer` to create an MCP server
- Running with different transports (stdio, HTTP)
- Exposing Gymnasium environments as MCP tools

**How to run:**
```bash
# Run with stdio transport (for Claude Desktop, etc.)
python examples/mcp_server_example.py

# Run with HTTP transport
python examples/mcp_server_example.py --transport streamable-http --port 8000

# Use a different environment
python examples/mcp_server_example.py --env MountainCar-v0
```

### 2. HTTP Transport Example (`http_example.py`)

Connect to an MCP server via HTTP transport and run episodes.

**What it demonstrates:**
- Starting a local MCP server in non-blocking mode (background thread)
- Connecting to external HTTP servers
- Using the official MCP Python SDK client (`ClientSession` with `streamablehttp_client`)
- Running multiple episodes programmatically
- Proper server lifecycle management and cleanup

**How to run:**
```bash
# Start local server and connect automatically (easiest):
python examples/http_example.py

# Use a different environment:
python examples/http_example.py --env MountainCar-v0

# Connect to external server:
python examples/http_example.py --external --host localhost --port 8765

# Run more episodes:
python examples/http_example.py --episodes 5
```

### 3. Stdio Transport Example (`stdio_example.py`)

Connect to an MCP server via stdio transport and run episodes.

**What it demonstrates:**
- Using stdio transport to launch and connect to the gym-mcp-server
- Using the official MCP Python SDK client (`ClientSession` with `stdio_client`)
- Running multiple episodes programmatically

**How to run:**
```bash
# Run with default environment:
python examples/stdio_example.py

# Use a different environment:
python examples/stdio_example.py --env MountainCar-v0

# Run more episodes:
python examples/stdio_example.py --episodes 5
```

### 4. OpenAI Agents SDK - stdio (`openai_agents_stdio_example.py`)

AI agent using OpenAI Agents SDK with stdio transport to autonomously control gym environments.

**What it demonstrates:**
- Using `MCPServerStdio` to launch the gym-mcp-server
- Creating agents with `mcp_servers` parameter
- Letting agents autonomously play episodes

**Requirements:**
```bash
pip install gym-mcp-server[dev]  # Includes openai-agents-sdk
export OPENAI_API_KEY=your_api_key_here
```

**How to run:**
```bash
python examples/openai_agents_stdio_example.py
```

### 5. OpenAI Agents SDK - HTTP (`openai_agents_http_example.py`)

AI agent using OpenAI Agents SDK with HTTP transport (for remote server connections).

**What it demonstrates:**
- Using `MCPServerStreamableHttp` to connect to a running server
- HTTP-based transport instead of stdio
- Suitable for remote/distributed setups

**Requirements:**
```bash
pip install gym-mcp-server[dev]  # Includes openai-agents-sdk
export OPENAI_API_KEY=your_api_key_here

# Start the server in a separate terminal first:
python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --port 8000
```

**How to run:**
```bash
python examples/openai_agents_http_example.py
```

## Documentation

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [OpenAI Agents SDK - MCP Integration](https://openai.github.io/openai-agents-python/mcp/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## Creating Your Own Examples

You can use any of these examples as a starting point for your own gym environment experiments:

1. **For HTTP transport client:** Use `http_example.py` as a template
2. **For stdio transport client:** Use `stdio_example.py` as a template
3. **For AI agent control:** Use `openai_agents_stdio_example.py` as a template

Simply replace `CartPole-v1` with any other Gymnasium environment ID (e.g., `MountainCar-v0`, `LunarLander-v2`, etc.).
