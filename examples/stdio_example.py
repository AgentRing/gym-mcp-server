#!/usr/bin/env python3
"""
Stdio transport example for Gymnasium MCP server.

This demonstrates running episodes using the stdio transport.

Requirements:
    pip install gym-mcp-server mcp

Usage:
    # Run with default CartPole environment:
    python stdio_example.py

    # Use a different environment:
    python stdio_example.py --env MountainCar-v0

    # Run more episodes:
    python stdio_example.py --episodes 5
"""

import argparse
import asyncio
import json
import logging
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_episode_stdio(
    env_id: str,
    num_episodes: int = 3,
):
    """
    Run episodes using stdio transport.

    Args:
        env_id: Environment ID
        num_episodes: Number of episodes to run
    """
    # Configure server parameters for stdio transport
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "gym_mcp_server", "--env", env_id],
    )

    logger.info(f"Starting MCP server for {env_id} via stdio...")

    async with stdio_client(server_params) as (read, write):
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Connect to a Gymnasium MCP server via stdio and run episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default environment:
  python stdio_example.py

  # Use a different environment:
  python stdio_example.py --env MountainCar-v0

  # Run more episodes:
  python stdio_example.py --episodes 5
        """,
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
            run_episode_stdio(
                env_id=args.env,
                num_episodes=args.episodes,
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
