"""
Entrypoint for running the gym_mcp_server as a module.
"""

import sys
import argparse
import logging
from .servers import GymHTTPServer
from .utils import register_env_from_entry_point

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for the combined HTTP (REST + MCP) server."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a Gymnasium environment over HTTP, "
            "exposing both REST and MCP (/mcp)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start combined REST + MCP server
  python -m gym_mcp_server --env CartPole-v1

  # Use a custom class via entry point
  python -m gym_mcp_server --entry-point mymodule.envs:MyEnv

  # Entry point with custom env ID
  python -m gym_mcp_server --entry-point mymodule.envs:MyEnv --env MyEnv-v0

  # Then open:
  # - REST docs: http://localhost:8000/docs
  # - MCP endpoint (streamable-http): http://localhost:8000/mcp
        """,
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help=(
            "Gymnasium environment ID (e.g., CartPole-v1). "
            "Required unless --entry-point is provided."
        ),
    )
    parser.add_argument(
        "--entry-point",
        type=str,
        default=None,
        dest="entry_point",
        help=(
            "Entry point in format 'module.path:ClassName' "
            "to dynamically register an environment. "
            "If --env is not provided, an ID will be auto-generated."
        ),
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Default render mode (e.g., rgb_array, human)",
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
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Title for the Swagger UI (default: derived from environment ID)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Description/subtitle for the Swagger UI",
    )
    parser.add_argument(
        "--enable-run-manager",
        action="store_true",
        default=True,
        help="Enable run manager for lifecycle tracking (default: True)",
    )
    parser.add_argument(
        "--disable-run-manager",
        action="store_true",
        default=False,
        help="Disable run manager for lifecycle tracking",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes in a run (default: 10)",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)",
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

        logger.info(f"Creating combined HTTP server for environment: {env_id}")

        # Determine if run manager should be enabled
        enable_run_manager = args.enable_run_manager and not args.disable_run_manager

        http_server = GymHTTPServer(
            env_id=env_id,
            render_mode=args.render_mode,
            title=args.title,
            description=args.description,
            enable_run_manager=enable_run_manager,
            num_episodes=args.num_episodes,
            max_steps_per_episode=args.max_steps_per_episode,
        )

        logger.info("Starting HTTP server (REST + MCP)...")
        # Banner will be printed automatically on server startup via FastAPI event

        http_server.run(host=args.host, port=args.port)

    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
