"""
HTTP endpoint definitions for gym environment operations.

This module contains all FastAPI endpoint handlers for the gym environment API.
"""

import logging
from typing import Any, Dict

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from .models import RenderRequest, ResetRequest, StepRequest
from .service import GymService

logger = logging.getLogger(__name__)


def _generate_index_html(
    env_id: str, base_url: str, max_episode_steps: int | None = None
) -> str:
    """Generate HTML for the index page.

    Args:
        env_id: The Gymnasium environment ID
        base_url: Base URL of the server
        max_episode_steps: Maximum number of steps per episode (optional)

    Returns:
        HTML string
    """
    from pathlib import Path
    from jinja2 import Environment, FileSystemLoader

    # Get the templates directory path
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("index.html")

    return template.render(
        env_id=env_id,
        base_url=base_url,
        max_episode_steps=max_episode_steps,
    )


def register_endpoints(app: FastAPI, service: GymService, env_id: str) -> None:
    """Register all endpoints on the FastAPI app.

    Args:
        app: FastAPI application instance
        service: GymService instance
        env_id: The Gymnasium environment ID
    """

    @app.post(
        "/reset", response_model=Dict[str, Any], operation_id="reset_env", tags=["gym"]
    )
    async def reset(
        request: ResetRequest = Body(default_factory=ResetRequest),
    ) -> Dict[str, Any]:
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

    @app.post(
        "/step", response_model=Dict[str, Any], operation_id="step_env", tags=["gym"]
    )
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

    @app.post(
        "/render",
        response_model=Dict[str, Any],
        operation_id="render_env",
        tags=["gym"],
    )
    async def render(
        request: RenderRequest = Body(default_factory=RenderRequest),
    ) -> Dict[str, Any]:
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

    @app.get(
        "/info",
        response_model=Dict[str, Any],
        operation_id="get_env_info",
        tags=["gym"],
    )
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

    @app.get(
        "/sample",
        response_model=Dict[str, Any],
        operation_id="sample_action",
        tags=["gym"],
    )
    async def sample() -> Dict[str, Any]:
        """Sample a random action from the environment's action space.

        Returns:
            Dictionary with sampled action and success status
        """
        try:
            result = service.sample()
            if not result.get("success", False):
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Failed to sample action"),
                )
            return result
        except Exception as e:
            logger.error(f"Error in sample endpoint: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/close", response_model=Dict[str, Any], operation_id="close_env", tags=["gym"]
    )
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

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def root(request: Request) -> HTMLResponse:
        """Root endpoint providing server information as HTML.

        Returns:
            HTML page with server information and available endpoints
        """
        # Get base URL from request
        base_url = str(request.base_url).rstrip("/")
        # Get max_episode_steps from environment info
        max_episode_steps = None
        if hasattr(app.state, "service"):
            env_info_result = app.state.service.get_info()
            if env_info_result.get("success") and "env_info" in env_info_result:
                max_episode_steps = env_info_result["env_info"].get("max_episode_steps")
        html_content = _generate_index_html(
            env_id=env_id, base_url=base_url, max_episode_steps=max_episode_steps
        )
        return HTMLResponse(content=html_content)

    @app.get("/api", response_model=Dict[str, Any], include_in_schema=False)
    async def api_info() -> Dict[str, Any]:
        """API endpoint providing server information as JSON.

        Returns:
            Dictionary with server information and available endpoints
        """
        # Get max_episode_steps from environment info
        max_episode_steps = None
        if hasattr(app.state, "service"):
            env_info_result = app.state.service.get_info()
            if env_info_result.get("success") and "env_info" in env_info_result:
                max_episode_steps = env_info_result["env_info"].get("max_episode_steps")

        return {
            "name": "Gym MCP Server",
            "version": "0.3.0",
            "environment": env_id,
            "max_episode_steps": max_episode_steps,
            "status": "running",
            "endpoints": {
                "rest": {
                    "reset": "POST /reset - Reset the environment to initial state",
                    "step": "POST /step - Take an action in the environment",
                    "render": "POST /render - Render the current state",
                    "info": "GET /info - Get environment information",
                    "sample": "GET /sample - Sample a random action from action space",
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

    @app.get("/health", include_in_schema=False)
    async def health() -> Dict[str, str]:
        """Health check endpoint.

        Returns:
            Dictionary with health status
        """
        return {"status": "healthy", "env_id": env_id}

    # Run manager endpoints
    @app.get(
        "/run/status",
        response_model=Dict[str, Any],
        operation_id="get_run_status",
        tags=["run"],
    )
    async def get_run_status() -> Dict[str, Any]:
        """Get current run status and progress.

        Returns:
            Dictionary with run status information
        """
        if not service.run_manager:
            raise HTTPException(
                status_code=404,
                detail="Run manager not enabled for this server",
            )
        return service.run_manager.get_status()

    @app.get(
        "/run/statistics",
        response_model=Dict[str, Any],
        operation_id="get_run_statistics",
        tags=["run"],
    )
    async def get_run_statistics() -> Dict[str, Any]:
        """Get complete run statistics.

        Returns:
            Dictionary with all run statistics
        """
        if not service.run_manager:
            raise HTTPException(
                status_code=404,
                detail="Run manager not enabled for this server",
            )
        return service.run_manager.get_statistics()

    @app.post(
        "/run/reset",
        response_model=Dict[str, Any],
        operation_id="reset_run",
        tags=["run"],
    )
    async def reset_run() -> Dict[str, Any]:
        """Reset the run manager for a new run.

        Returns:
            Dictionary with reset confirmation
        """
        if not service.run_manager:
            raise HTTPException(
                status_code=404,
                detail="Run manager not enabled for this server",
            )
        service.run_manager.reset_run()
        service.run_manager.start_run()
        return {
            "success": True,
            "message": "Run reset successfully",
            "run_id": service.run_manager.run_id,
        }
