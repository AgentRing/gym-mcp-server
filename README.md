# Gym MCP Server

Expose any Gymnasium environment as an MCP (Model Context Protocol) server, automatically converting the Gym API (`reset`, `step`, `render`) into MCP tools that any agent can call via standard JSON interfaces.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test Coverage](https://img.shields.io/badge/coverage-60%25-brightgreen.svg)](htmlcov/index.html)

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
- **`sample_action`** - Sample a random action from the action space

All tools return a standardized format:

```python
{
    "success": bool,  # Whether the operation succeeded
    "error": str,     # Error message (if success=False)
    # ... tool-specific data
}
```

## Examples

You can use the server with any MCP-compatible client. Here's a simple example using the MCP Python client:

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

async with streamable_http_client("http://localhost:8000/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        print(f"Available tools: {[tool.name for tool in tools.tools]}")
        
        # Reset the environment
        result = await session.call_tool("reset_env", arguments={})
        print(f"Reset result: {result.content[0].text}")
```

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

## Run Management Lifecycle

The server includes an optional **Run Manager** that tracks the lifecycle of training runs, episodes, and steps. This is enabled by default and provides comprehensive statistics and progress tracking.

### Overview

A **run** consists of multiple **episodes**, where each episode is a complete interaction sequence from reset to termination. The run manager automatically tracks:

- Run state and progress
- Episode statistics (steps, rewards, duration)
- Step-by-step details
- Aggregated metrics (average reward, success rate, etc.)

### Run States

The run progresses through these states:

- **`IDLE`** - No active run (initial state)
- **`RUNNING`** - Run in progress, accepting episodes
- **`COMPLETED`** - All episodes finished successfully
- **`ERROR`** - Run failed due to an error
- **`PAUSED`** - Run paused (future feature)

### Lifecycle Flow

1. **Run Initialization**
   - When `GymService` is created with `enable_run_manager=True` (default), a run manager is initialized
   - The run automatically starts in `RUNNING` state
   - Configuration: `num_episodes` (default: 10), `max_steps_per_episode` (default: 1000)

2. **Episode Lifecycle**
   - Each call to `reset_env()` starts a new episode
   - The run manager tracks episode number, start time, and seed
   - Episodes are numbered sequentially (1, 2, 3, ...)

3. **Step Tracking**
   - Each call to `step_env()` records a step in the current episode
   - The manager tracks: action, observation, reward, done, truncated, info
   - Step limits are enforced per episode (`max_steps_per_episode`)

4. **Episode Completion**
   - An episode completes when `done=True` or `truncated=True` from `step_env()`
   - Episode statistics are finalized: total steps, total reward, duration, success status
   - The episode is added to the run's episode history

5. **Run Completion**
   - When all configured episodes are completed, the run transitions to `COMPLETED` state
   - Final statistics are calculated: average reward, success rate, total duration
   - The run manager stops accepting new episodes

6. **Run Reset**
   - Use `POST /run/reset` to start a new run with the same configuration
   - All statistics are cleared and a new run ID is generated
   - The run state returns to `IDLE`, then automatically starts in `RUNNING`

### API Endpoints

The run manager exposes these REST endpoints:

#### Get Run Status

```bash
GET /run/status
```

Returns current run progress:

```json
{
  "run_id": "uuid-string",
  "state": "running",
  "current_episode": 3,
  "current_step": 42,
  "completed_episodes": 2,
  "total_episodes": 10,
  "progress": {
    "episodes": "2/10",
    "percentage": 20.0
  }
}
```

#### Get Run Statistics

```bash
GET /run/statistics
```

Returns comprehensive statistics for the entire run:

```json
{
  "run_id": "uuid-string",
  "env_id": "CartPole-v1",
  "num_episodes": 10,
  "max_steps_per_episode": 1000,
  "state": "running",
  "start_time": "2024-01-01T12:00:00",
  "end_time": null,
  "current_episode": 3,
  "current_step": 42,
  "completed_episodes": 2,
  "total_steps": 150,
  "total_reward": 450.0,
  "average_reward": 225.0,
  "average_steps": 75.0,
  "success_rate": 1.0,
  "successful_episodes": 2,
  "duration": 12.5,
  "episodes": [
    {
      "episode_num": 1,
      "total_steps": 80,
      "total_reward": 200.0,
      "final_reward": 1.0,
      "done": true,
      "truncated": false,
      "success": true,
      "duration": 5.2
    },
    {
      "episode_num": 2,
      "total_steps": 70,
      "total_reward": 250.0,
      "final_reward": 1.0,
      "done": true,
      "truncated": false,
      "success": true,
      "duration": 4.8
    }
  ]
}
```

#### Reset Run

```bash
POST /run/reset
```

Resets the run manager to start a new run:

```json
{
  "success": true,
  "message": "Run reset successfully",
  "run_id": "new-uuid-string"
}
```

### Response Integration

The run manager automatically adds progress information to standard tool responses:

**`reset_env` response includes:**
```json
{
  "observation": {...},
  "info": {...},
  "success": true,
  "run_progress": {
    "episode_num": 1,
    "total_episodes": 10,
    "max_steps": 1000
  }
}
```

**`step_env` response includes:**
```json
{
  "observation": {...},
  "reward": 1.0,
  "done": false,
  "success": true,
  "run_progress": {
    "completed_episodes": 2,
    "total_episodes": 10,
    "run_state": "running"
  }
}
```

### Configuration Options

When creating `GymService` programmatically:

```python
from gym_mcp_server import GymService

# Enable run manager with custom settings
service = GymService(
    env_id="CartPole-v1",
    enable_run_manager=True,      # Enable run tracking (default: True)
    num_episodes=20,               # Number of episodes in run (default: 10)
    max_steps_per_episode=500,    # Max steps per episode (default: 1000)
)

# Disable run manager
service = GymService(
    env_id="CartPole-v1",
    enable_run_manager=False,      # Disable run tracking
)
```

### Best Practices

1. **Monitor Progress**: Regularly check `/run/status` to track progress during long runs
2. **Collect Statistics**: Use `/run/statistics` after completion to analyze performance
3. **Reset Between Experiments**: Use `/run/reset` to start fresh runs with the same configuration
4. **Handle Limits**: Be aware of `max_steps_per_episode` - episodes exceeding this limit are automatically truncated
5. **Episode Completion**: Ensure episodes complete naturally (via `done=True`) rather than always hitting step limits

### Disabling Run Manager

If you don't need run tracking, you can disable it:

```python
from gym_mcp_server import GymHTTPServer

server = GymHTTPServer(
    env_id="CartPole-v1",
    enable_run_manager=False  # Disable run tracking
)
```

When disabled, the run management endpoints (`/run/*`) will return 404 errors.

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
make check       # Run all checks (lint, typecheck, test)
```

See the [Makefile](Makefile) for all available commands.

## License

MIT License - see the LICENSE file for details.

## Links

- [GitHub Repository](https://github.com/haggaishachar/gym-mcp-server)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Model Context Protocol](https://modelcontextprotocol.io/)

