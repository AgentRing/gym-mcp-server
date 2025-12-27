"""
HTTP server implementation for gym environment operations.

This module provides FastAPI-based REST endpoints for the GymService.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from jinja2 import Environment, FileSystemLoader

from .endpoints import register_endpoints
from .service import GymService

logger = logging.getLogger(__name__)


def _print_server_banner(
    env_id: str, host: str, port: int, max_episode_steps: Optional[int] = None
) -> None:
    """Print a professional server startup banner with server information.

    Args:
        env_id: The Gymnasium environment ID
        host: Server host
        port: Server port
        max_episode_steps: Maximum number of steps per episode (optional)
    """
    base_url = f"http://{host}:{port}"
    # Get the templates directory path
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("banner.txt")

    banner = template.render(
        env_id=env_id,
        base_url=base_url,
        max_episode_steps=max_episode_steps,
    )
    print(banner)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI app."""
    # Startup
    if (
        hasattr(app.state, "host")
        and hasattr(app.state, "port")
        and app.state.host is not None
        and app.state.port is not None
    ):
        # Get max_episode_steps from environment info
        max_episode_steps = None
        if hasattr(app.state, "service"):
            env_info_result = app.state.service.get_info()
            if env_info_result.get("success") and "env_info" in env_info_result:
                max_episode_steps = env_info_result["env_info"].get("max_episode_steps")

        _print_server_banner(
            env_id=app.state.env_id,
            host=app.state.host,
            port=app.state.port,
            max_episode_steps=max_episode_steps,
        )

    yield

    # Shutdown (if needed in the future)
    pass


def create_app(
    env_id: str,
    render_mode: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> FastAPI:
    """Create a FastAPI application for the gym environment.

    Args:
        env_id: The Gymnasium environment ID
        render_mode: Optional render mode for the environment
        title: Optional title for the Swagger UI
        description: Optional description for the Swagger UI
        host: Server host (for banner display)
        port: Server port (for banner display)

    Returns:
        FastAPI application instance
    """
    app_title = title or f"Gym Environment API - {env_id}"
    app_description = (
        description or f"REST API for interacting with {env_id} environment"
    )

    app = FastAPI(
        title=app_title,
        description=app_description,
        version="0.3.0",
        lifespan=lifespan,
    )

    # Store host and port for banner display
    app.state.host = host
    app.state.port = port
    app.state.env_id = env_id

    # Initialize the service
    service = GymService(env_id=env_id, render_mode=render_mode)
    # Store service in app state for access in endpoints and startup
    app.state.service = service

    # Register all endpoints
    register_endpoints(app, service, env_id)

    # Add MCP support
    try:
        # Create MCP server with only gym-tagged endpoints
        mcp = FastApiMCP(
            app,
            include_tags=["gym"],  # Only include endpoints tagged with "gym"
        )
        # Mount both HTTP (streamable-http) and SSE transports
        mcp.mount_http()  # HTTP transport at /mcp
        mcp.mount_sse()  # SSE transport at /sse
        logger.info("MCP server mounted at /mcp (HTTP) and /sse (SSE) with gym tools")
    except Exception as e:
        logger.warning(f"Failed to mount MCP server: {e}")

    return app


class GymHTTPServer:
    """HTTP server wrapper for gym environment operations.

    This class provides a simple interface to run a FastAPI server
    that exposes the GymService as REST endpoints.
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the HTTP server.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Optional render mode for the environment
            title: Optional title for the Swagger UI
            description: Optional description for the Swagger UI
        """
        self.env_id = env_id
        self.render_mode = render_mode
        self.title = title
        self.description = description
        # Create app without host/port initially (will be updated in run())
        self.app = create_app(
            env_id=env_id,
            render_mode=render_mode,
            title=title,
            description=description,
            host=None,
            port=None,
        )

    def run(self, host: str = "localhost", port: int = 8000) -> None:
        """Run the HTTP server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        import uvicorn

        # Update app state with host/port for banner display
        self.app.state.host = host
        self.app.state.port = port

        uvicorn.run(self.app, host=host, port=port)
