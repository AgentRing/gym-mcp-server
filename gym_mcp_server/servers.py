"""
HTTP server implementation for gym environment operations.

This module provides FastAPI-based REST endpoints for the GymService.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from fastapi_mcp import FastApiMCP
except ImportError:
    FastApiMCP = None

from .service import GymService

logger = logging.getLogger(__name__)


# Request/Response models
class ResetRequest(BaseModel):
    """Request model for reset endpoint."""

    seed: Optional[int] = None


class StepRequest(BaseModel):
    """Request model for step endpoint."""

    action: Any


class RenderRequest(BaseModel):
    """Request model for render endpoint."""

    mode: Optional[str] = None


def create_app(
    env_id: str,
    render_mode: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
) -> FastAPI:
    """Create a FastAPI application for the gym environment.

    Args:
        env_id: The Gymnasium environment ID
        render_mode: Optional render mode for the environment
        title: Optional title for the Swagger UI
        description: Optional description for the Swagger UI

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
    )

    # Initialize the service
    service = GymService(env_id=env_id, render_mode=render_mode)

    @app.post("/reset", response_model=Dict[str, Any])
    async def reset(request: ResetRequest) -> Dict[str, Any]:
        """Reset the environment to its initial state.

        Args:
            request: Reset request with optional seed

        Returns:
            Dictionary with observation, info, done, and success status
        """
        try:
            result = service.reset(seed=request.seed)
            if not result.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to reset environment"),
                )
            return result
        except Exception as e:
            logger.error(f"Error in reset endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/step", response_model=Dict[str, Any])
    async def step(request: StepRequest) -> Dict[str, Any]:
        """Take an action in the environment.

        Args:
            request: Step request with action

        Returns:
            Dictionary with observation, reward, done, truncated, info,
            and success status
        """
        try:
            result = service.step(action=request.action)
            if not result.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to take step"),
                )
            return result
        except Exception as e:
            logger.error(f"Error in step endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/render", response_model=Dict[str, Any])
    async def render(request: RenderRequest) -> Dict[str, Any]:
        """Render the current state of the environment.

        Args:
            request: Render request with optional mode

        Returns:
            Dictionary with render output, mode, type, shape, and success status
        """
        try:
            result = service.render(mode=request.mode)
            if not result.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to render environment"),
                )
            return result
        except Exception as e:
            logger.error(f"Error in render endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/info", response_model=Dict[str, Any])
    async def get_info() -> Dict[str, Any]:
        """Get information about the environment.

        Returns:
            Dictionary with environment metadata and success status
        """
        try:
            result = service.get_info()
            if not result.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to get environment info"),
                )
            return result
        except Exception as e:
            logger.error(f"Error in info endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/close", response_model=Dict[str, Any])
    async def close() -> Dict[str, Any]:
        """Close the environment and free resources.

        Returns:
            Dictionary with close status and success flag
        """
        try:
            result = service.close()
            if not result.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to close environment"),
                )
            return result
        except Exception as e:
            logger.error(f"Error in close endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root() -> Dict[str, Any]:
        """Root endpoint providing server information.

        Returns:
            Dictionary with server information and available endpoints
        """
        return {
            "name": "Gym MCP Server",
            "version": "0.3.0",
            "environment": env_id,
            "status": "running",
            "endpoints": {
                "rest": {
                    "reset": "POST /reset - Reset the environment to initial state",
                    "step": "POST /step - Take an action in the environment",
                    "render": "POST /render - Render the current state",
                    "info": "GET /info - Get environment information",
                    "close": "POST /close - Close the environment",
                    "health": "GET /health - Health check",
                },
                "mcp": {
                    "mcp": "POST /mcp - MCP endpoint (streamable-http transport)",
                },
                "docs": {
                    "swagger": "GET /docs - Swagger UI documentation",
                    "openapi": "GET /openapi.json - OpenAPI schema",
                },
            },
            "links": {
                "swagger_ui": "/docs",
                "openapi_schema": "/openapi.json",
                "mcp_endpoint": "/mcp",
            },
        }

    @app.get("/health")
    async def health() -> Dict[str, str]:
        """Health check endpoint.

        Returns:
            Dictionary with health status
        """
        return {"status": "healthy", "env_id": env_id}

    # Add MCP support if fastapi-mcp is available
    if FastApiMCP is not None:
        try:
            mcp = FastApiMCP(app)
            mcp.mount_http()
            logger.info("MCP server mounted at /mcp")
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
        self.app = create_app(
            env_id=env_id,
            render_mode=render_mode,
            title=title,
            description=description,
        )

    def run(self, host: str = "localhost", port: int = 8000) -> None:
        """Run the HTTP server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)
