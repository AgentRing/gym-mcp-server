"""
Entrypoint for running the gym_mcp_server as a module.
"""

import sys
import argparse
import logging
from typing import Any, Dict, Optional
from .server import GymMCPServer
from .utils import register_env_from_entry_point

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="Run a Gymnasium environment as an MCP server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use a registered environment
  python -m gym_mcp_server --env CartPole-v1

  # Use a custom class via entry point
  python -m gym_mcp_server --entry-point mymodule.envs:MyEnv

  # Entry point with custom env ID
  python -m gym_mcp_server --entry-point mymodule.envs:MyEnv --env MyEnv-v0

  # HTTP transport
  python -m gym_mcp_server --env CartPole-v1 --transport streamable-http --port 8000
        """,
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Gymnasium environment ID (e.g., CartPole-v1). Required unless --entry-point is provided.",
    )
    parser.add_argument(
        "--entry-point",
        type=str,
        default=None,
        dest="entry_point",
        help="Entry point in format 'module.path:ClassName' to dynamically register an environment. "
        "If --env is not provided, an ID will be auto-generated.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Default render mode (e.g., rgb_array, human)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "streamable-http", "sse"],
        help="Transport type for the MCP server (default: stdio)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for HTTP-based transports (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP-based transports (default: 8000)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.env is None and args.entry_point is None:
        parser.error("Either --env or --entry-point must be provided")

    try:
        # Handle entry point registration
        env_id: str
        if args.entry_point:
            # Register environment from entry point
            env_id = register_env_from_entry_point(
                entry_point=args.entry_point,
                env_id=args.env,  # Use provided env ID or auto-generate
            )
            logger.info(f"Registered environment from entry point: {env_id}")
        else:
            env_id = args.env

        # Create and run the MCP server
        logger.info(f"Creating MCP server for environment: {env_id}")
        logger.info(f"Transport: {args.transport}")

        # Build kwargs for GymMCPServer based on transport
        kwargs: Dict[str, Any] = {}
        if args.transport in ("streamable-http", "sse"):
            kwargs["host"] = args.host
            kwargs["port"] = args.port
            logger.info(f"Server will listen on {args.host}:{args.port}")

        gym_server = GymMCPServer(
            env_id=env_id,
            render_mode=args.render_mode,
            **kwargs,
        )

        # Run the server with the specified transport
        logger.info("Starting MCP server...")
        logger.info(
            "The server exposes the following MCP tools:\n"
            "  - reset_env: Reset the environment\n"
            "  - step_env: Take an action in the environment\n"
            "  - render_env: Render the environment\n"
            "  - get_env_info: Get environment information\n"
            "  - close_env: Close the environment"
        )

        if args.transport == "stdio":
            logger.info(
                "Server running on stdio. Connect using an MCP client "
                "(e.g., Claude Desktop, MCP Inspector)."
            )
        else:
            logger.info(f"Server ready at http://{args.host}:{args.port}")

        # Run the server
        gym_server.run(transport=args.transport)

    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
