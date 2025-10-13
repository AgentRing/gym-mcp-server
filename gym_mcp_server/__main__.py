"""
Entrypoint for running the gym_mcp_server as a module.
"""

import sys
import argparse
import logging
from typing import Any, Dict
from .server import GymMCPServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for the MCP server."""
    parser = argparse.ArgumentParser(
        description="Run a Gymnasium environment as an MCP server."
    )
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Gymnasium environment ID (e.g., CartPole-v1)",
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

    try:
        # Create and run the MCP server
        logger.info(f"Creating MCP server for environment: {args.env}")
        logger.info(f"Transport: {args.transport}")

        # Build kwargs for GymMCPServer based on transport
        kwargs: Dict[str, Any] = {}
        if args.transport in ("streamable-http", "sse"):
            kwargs["host"] = args.host
            kwargs["port"] = args.port
            logger.info(f"Server will listen on {args.host}:{args.port}")

        gym_server = GymMCPServer(
            env_id=args.env,
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
