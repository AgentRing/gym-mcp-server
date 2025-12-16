"""
FastAPI HTTP server with Swagger UI for interacting with Gymnasium environments.

This module provides a REST API alternative to the MCP server, with automatic
Swagger UI documentation at /docs.
"""

import gymnasium as gym
import logging
import threading
import time
import uvicorn
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .utils import (
    serialize_observation,
    serialize_render_output,
    get_environment_info,
    register_env_from_entry_point,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request/response validation
class ResetRequest(BaseModel):
    """Request model for resetting the environment."""

    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducible episodes"
    )


class ResetResponse(BaseModel):
    """Response model for environment reset."""

    observation: Any = Field(description="Initial observation from the environment")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional information from the environment"
    )
    done: bool = Field(
        default=False, description="Whether the episode is done (always False for reset)"
    )
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class StepRequest(BaseModel):
    """Request model for taking a step in the environment."""

    action: Any = Field(description="Action to take in the environment")


class StepResponse(BaseModel):
    """Response model for environment step."""

    observation: Any = Field(description="Next observation from the environment")
    reward: float = Field(description="Reward received from the action")
    done: bool = Field(description="Whether the episode is done")
    truncated: bool = Field(default=False, description="Whether the episode was truncated")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional information from the environment"
    )
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RenderRequest(BaseModel):
    """Request model for rendering the environment."""

    mode: Optional[str] = Field(
        default=None, description="Render mode (e.g., rgb_array, human)"
    )


class RenderResponse(BaseModel):
    """Response model for environment render."""

    render: Any = Field(description="Rendered output from the environment")
    mode: Optional[str] = Field(description="Render mode used")
    type: Optional[str] = Field(
        default=None,
        description="Type of render output (text, image_base64, array, string)",
    )
    shape: Optional[List[int]] = Field(
        default=None, description="Shape of the render output (for arrays/images)"
    )
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class EnvInfoResponse(BaseModel):
    """Response model for environment info."""

    env_info: Dict[str, Any] = Field(description="Environment metadata")
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class CloseResponse(BaseModel):
    """Response model for closing the environment."""

    status: str = Field(description="Status of the close operation")
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(description="Server status")
    env_id: str = Field(description="Currently loaded environment ID")
    render_mode: Optional[str] = Field(description="Current render mode")


class GymHTTPServer:
    """HTTP Server with Swagger UI for Gymnasium environments.

    This class provides a FastAPI-based REST API for interacting with
    Gymnasium environments through HTTP endpoints with automatic Swagger UI.

    Example (blocking):
        >>> server = GymHTTPServer(env_id="CartPole-v1")
        >>> server.run(host="localhost", port=8000)

    Example (non-blocking):
        >>> server = GymHTTPServer(env_id="CartPole-v1")
        >>> server.start(host="localhost", port=8000)  # Start in background
        >>> # ... do other work ...
        >>> server.stop()  # Stop the server
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """Initialize the Gym HTTP Server.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Optional render mode for the environment
            title: Title for the Swagger UI (default: derived from env_id)
            description: Description/subtitle for the Swagger UI
        """
        self.env_id = env_id
        self.render_mode = render_mode

        # Derive a meaningful title from env_id if not provided
        if title is None:
            # Extract a clean name from the env_id (e.g., "CartPole-v1" -> "CartPole")
            # Handle dynamic IDs like "WebAgentTextEnv-dynamic-eb33bd93-v0"
            base_name = env_id.split("-dynamic-")[0].split("-v")[0]
            # Convert camelCase to Title Case with spaces
            import re
            title = re.sub(r"([a-z])([A-Z])", r"\1 \2", base_name)
            title = f"{title} Environment API"

        self.title = title
        self.description = description

        # Initialize the Gymnasium environment
        logger.info(f"Initializing environment: {env_id}")
        if render_mode is not None:
            self.env = gym.make(env_id, render_mode=render_mode)
        else:
            self.env = gym.make(env_id)

        # Create FastAPI app
        self._app = self._create_app(title, description)

        # Threading support for non-blocking mode
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    def _create_app(self, title: str, description: Optional[str] = None) -> FastAPI:
        """Create and configure the FastAPI application."""
        # Build the full description with optional custom subtitle
        subtitle = f"\n\n{description}\n" if description else ""
        full_description = f"""## {title}
{subtitle}
Interact with the **{self.env_id}** environment through REST endpoints.

### Available Operations:
- **Reset**: Initialize or reset the environment
- **Step**: Take an action and observe the result
- **Render**: Get a visual representation of the current state
- **Info**: Get environment metadata (action/observation spaces)
- **Close**: Clean up environment resources

### Usage Flow:
1. Call `/reset` to initialize the environment
2. Call `/step` with actions to interact
3. Use `/render` to visualize the state
4. Call `/close` when done
"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):  # type: ignore[misc]
            # Startup
            logger.info(f"HTTP Server starting with environment: {self.env_id}")
            yield
            # Shutdown
            logger.info("HTTP Server shutting down...")
            try:
                self.env.close()
            except Exception as e:
                logger.error(f"Error closing environment on shutdown: {e}")

        app = FastAPI(
            title=title,
            description=full_description,
            version="0.3.0",
            lifespan=lifespan,
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json",
        )

        # Add CORS middleware for browser access
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Register routes
        self._register_routes(app)

        return app

    def _register_routes(self, app: FastAPI) -> None:
        """Register all API routes."""

        @app.get("/", tags=["Health"])
        async def root() -> Dict[str, str]:
            """Root endpoint with API information."""
            return {
                "message": "Gym Environment HTTP API",
                "docs": "/docs",
                "redoc": "/redoc",
                "env_id": self.env_id,
            }

        @app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health_check() -> HealthResponse:
            """Check server health and current environment status."""
            return HealthResponse(
                status="healthy",
                env_id=self.env_id,
                render_mode=self.render_mode,
            )

        @app.post("/reset", response_model=ResetResponse, tags=["Environment"])
        async def reset_env(request: ResetRequest = ResetRequest()) -> ResetResponse:
            """Reset the environment to its initial state.

            Optionally provide a seed for reproducible episodes.
            """
            logger.info(f"Resetting environment with seed={request.seed}")
            try:
                if request.seed is not None:
                    obs, info = self.env.reset(seed=request.seed)
                else:
                    obs, info = self.env.reset()

                return ResetResponse(
                    observation=serialize_observation(obs),
                    info=info,
                    done=False,
                    success=True,
                )
            except Exception as e:
                logger.error(f"Error resetting environment: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/step", response_model=StepResponse, tags=["Environment"])
        async def step_env(request: StepRequest) -> StepResponse:
            """Take an action in the environment.

            Provide the action appropriate for the environment's action space.
            For discrete spaces, use an integer. For continuous spaces, use a list.
            """
            logger.info(f"Taking step with action={request.action}")
            try:
                # Convert action to appropriate format if needed
                action: Any = request.action
                if isinstance(action, (list, tuple)) and len(action) == 1:
                    action = action[0]

                obs, reward, done, truncated, info = self.env.step(action)

                return StepResponse(
                    observation=serialize_observation(obs),
                    reward=float(reward),
                    done=bool(done or truncated),
                    truncated=bool(truncated),
                    info=info,
                    success=True,
                )
            except Exception as e:
                logger.error(f"Error taking step: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/render", response_model=RenderResponse, tags=["Environment"])
        async def render_env(
            request: RenderRequest = RenderRequest(),
        ) -> RenderResponse:
            """Render the current state of the environment.

            For rgb_array mode, returns a base64-encoded PNG image.
            """
            logger.info(f"Rendering environment with mode={request.mode}")
            try:
                render_out: Any = self.env.render()
                result = serialize_render_output(
                    render_out, request.mode or self.render_mode
                )
                return RenderResponse(
                    render=result.get("render"),
                    mode=result.get("mode"),
                    type=result.get("type"),
                    shape=result.get("shape"),
                    success=True,
                )
            except Exception as e:
                logger.error(f"Error rendering environment: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/info", response_model=EnvInfoResponse, tags=["Environment"])
        async def get_env_info() -> EnvInfoResponse:
            """Get information about the environment.

            Returns metadata including action space, observation space,
            and reward range.
            """
            logger.info("Getting environment info")
            try:
                return EnvInfoResponse(
                    env_info=get_environment_info(self.env),
                    success=True,
                )
            except Exception as e:
                logger.error(f"Error getting environment info: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/close", response_model=CloseResponse, tags=["Environment"])
        async def close_env() -> CloseResponse:
            """Close the environment and free resources.

            Note: The environment will be automatically reopened on the next
            reset call.
            """
            logger.info("Closing environment")
            try:
                self.env.close()
                return CloseResponse(status="closed", success=True)
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application instance."""
        return self._app

    def run(self, host: str = "localhost", port: int = 8000) -> None:
        """Run the HTTP server (blocking).

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        logger.info(f"Starting HTTP server at http://{host}:{port}")
        logger.info(f"Swagger UI available at http://{host}:{port}/docs")
        logger.info(f"ReDoc available at http://{host}:{port}/redoc")
        uvicorn.run(self._app, host=host, port=port)

    def start(self, host: str = "localhost", port: int = 8000) -> None:
        """Start the server in a background thread (non-blocking).

        Args:
            host: Host to bind to
            port: Port to bind to

        Raises:
            RuntimeError: If server is already running
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Server already started")

        config = uvicorn.Config(self._app, host=host, port=port, log_level="info")
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        # Wait for server to start
        logger.info("Waiting for HTTP server to start...")
        time.sleep(1)

        logger.info(f"HTTP server started at http://{host}:{port}")
        logger.info(f"Swagger UI: http://{host}:{port}/docs")

    def stop(self) -> None:
        """Stop the server if running in non-blocking mode."""
        if self._server is not None:
            logger.info("Stopping HTTP server...")
            self._server.should_exit = True

    def is_alive(self) -> bool:
        """Check if the server thread is alive."""
        return self._thread is not None and self._thread.is_alive()


def create_app(
    env_id: str,
    render_mode: Optional[str] = None,
    entry_point: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> FastAPI:
    """Factory function to create a FastAPI app for a gym environment.

    This is useful for running with external ASGI servers like gunicorn.

    Args:
        env_id: The Gymnasium environment ID
        render_mode: Optional render mode
        entry_point: Optional entry point to register a custom environment
        title: Optional title for Swagger UI (default: derived from env_id)
        description: Optional description/subtitle for Swagger UI

    Returns:
        Configured FastAPI application

    Example:
        # Run with uvicorn directly:
        # uvicorn "gym_mcp_server.http_server:create_app" --factory --env CartPole-v1
    """
    # Handle entry point registration if provided
    if entry_point:
        env_id = register_env_from_entry_point(entry_point=entry_point, env_id=env_id)

    server = GymHTTPServer(
        env_id=env_id,
        render_mode=render_mode,
        title=title,
        description=description,
    )
    return server.app

