#!/usr/bin/env python3
"""
HTTP transport example for Gymnasium MCP server.

This demonstrates running episodes using the HTTP transport.

Requirements:
    pip install gym-mcp-server mcp

Usage:
    # Automatically start local server and connect (default):
    python http_example.py

    # Connect to an external HTTP server:
    python http_example.py --external --host localhost --port 8000

    # Use a different environment:
    python http_example.py --env MountainCar-v0
"""

import argparse
import asyncio
import json
import logging
from typing import Optional
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from gym_mcp_server import GymMCPServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_episode_http(
    host: str, port: int, env_id: str, num_episodes: int = 3, start_server: bool = True
):
    """
    Run episodes using HTTP transport.

    Args:
        host: Server host
        port: Server port
        env_id: Environment ID
        num_episodes: Number of episodes to run
        start_server: Whether to start a local server in non-blocking mode
    """
    server: Optional[GymMCPServer] = None

    try:
        # Start local server if requested
        if start_server:
            logger.info(f"Starting local MCP server for {env_id}...")
            server = GymMCPServer(
                env_id=env_id,
                render_mode=None,
                host=host,
                port=port,
            )
            server.start()

            # Verify server is running
            if not server.is_alive():
                raise RuntimeError("Failed to start local server")

        # Connect to the server
        url = f"http://{host}:{port}/mcp"
        logger.info(f"Connecting to MCP server at {url}...")

        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                logger.info("✓ Connected to MCP server")

                # List available tools
                tools_response = await session.list_tools()
                logger.info(
                    f"Available tools: {[tool.name for tool in tools_response.tools]}"
                )

                # Run episodes
                for episode in range(num_episodes):
                    logger.info(f"\n--- Episode {episode + 1} ---")

                    # Reset environment
                    reset_result = await session.call_tool("reset_env", arguments={})
                    reset_data = json.loads(reset_result.content[0].text)
                    logger.info(
                        f"Reset: observation shape = {len(reset_data['observation'])}"
                    )

                    # Run until done
                    done = False
                    total_reward = 0
                    steps = 0

                    while not done and steps < 100:
                        # Take random action (0 or 1 for CartPole)
                        action = steps % 2  # Alternate actions

                        step_result = await session.call_tool(
                            "step_env", arguments={"action": action}
                        )
                        step_data = json.loads(step_result.content[0].text)

                        total_reward += step_data["reward"]
                        done = step_data["done"]
                    steps += 1

                logger.info(
                    f"Episode {episode + 1}: {steps} steps, "
                    f"total reward: {total_reward:.2f}"
                )

                # Get environment info
                info_result = await session.call_tool("get_env_info", arguments={})
                info_data = json.loads(info_result.content[0].text)
                logger.info(f"\nEnvironment info: {info_data['env_info']['id']}")

                # Close environment
                await session.call_tool("close_env", arguments={})
                logger.info("Environment closed")

    finally:
        # Stop local server if we started it
        if server is not None:
            logger.info("Stopping local server...")
            server.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Connect to a Gymnasium MCP server via HTTP and run episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start local server and connect (easiest):
  python http_example.py

  # Use a different environment:
  python http_example.py --env MountainCar-v0

  # Connect to external server:
  python http_example.py --external --host localhost --port 8765

  # Run more episodes:
  python http_example.py --episodes 5
        """,
    )
    parser.add_argument(
        "--external",
        action="store_true",
        help="Connect to external server instead of starting local one",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port (default: 8765)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment ID (default: CartPole-v1)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run (default: 3)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            run_episode_http(
                host=args.host,
                port=args.port,
                env_id=args.env,
                num_episodes=args.episodes,
                start_server=not args.external,
            )
        )
        logger.info("\n✓ All episodes completed successfully!")
        return 0
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error running client: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
