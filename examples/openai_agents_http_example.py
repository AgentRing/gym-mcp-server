#!/usr/bin/env python3
"""
Example showing how to use the gym-mcp-server with OpenAI Agents SDK via HTTP transport.

This demonstrates connecting to a gym-mcp-server running as an HTTP server
instead of using stdio transport.

Requirements:
    pip install gym-mcp-server[dev]
    # Or install the OpenAI Agents SDK separately: pip install openai-agents-sdk

Setup:
    1. Start the server in a separate terminal:
       python -m gym_mcp_server --env CartPole-v1 \\
       --transport streamable-http --port 8000

    2. Run this script:
       python examples/openai_agents_http_example.py

Documentation:
    https://openai.github.io/openai-agents-python/mcp/
"""

import asyncio
import os

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp


async def main():
    """Main entry point."""
    try:
        # Check for OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY environment variable not set.")
            print("The agent will not be able to run without it.\n")

        print("=== OpenAI Agents SDK + Gym MCP Server (HTTP) Example ===\n")
        print(
            "Note: Make sure the gym-mcp-server is running at " "http://localhost:8000"
        )
        print(
            "      python -m gym_mcp_server --env CartPole-v1 "
            "--transport streamable-http --port 8000\n"
        )

        # Connect to the gym-mcp-server via HTTP transport
        async with MCPServerStreamableHttp(
            name="Gym Environment",
            params={
                "url": "http://localhost:8000/mcp",
                "timeout": 10,
            },
            cache_tools_list=True,
        ) as server:
            print("✓ Connected to gym-mcp-server via HTTP\n")

            # Create an agent with access to the gym environment tools
            agent = Agent(
                name="GymAgent",
                instructions="""You are an agent that controls CartPole-v1.

Your goal is to:
1. Reset the environment to start a new episode
2. Take actions to balance the pole on the cart
3. Try to maximize the total reward by keeping the pole balanced

The CartPole environment has:
- 4 observations: cart position, cart velocity, pole angle, pole angular velocity
- 2 actions: 0 (push left) or 1 (push right)
- Episode ends when pole falls too far or cart moves off screen

Use the available tools to interact with the environment.""",
                mcp_servers=[server],
            )

            print("Running agent to play one episode...\n")
            print("-" * 60)

            # Let the agent autonomously play the environment
            result = await Runner.run(
                agent,
                """Reset the environment and play one complete episode of CartPole.
            For each step, decide whether to push left (0) or right (1).
            Try to keep the pole balanced for as long as possible.
            Report the total reward at the end.""",
                max_turns=10,  # Allow up to 10 turns for a full CartPole episode
            )

            print("-" * 60)
            print("\nAgent Result:")
            print(result.final_output)
            print()

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
