"""
HTTP server implementation for gym environment operations.

This module provides FastAPI-based REST endpoints for the GymService.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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


def _print_server_banner(env_id: str, host: str, port: int) -> None:
    """Print a professional server startup banner with server information.

    Args:
        env_id: The Gymnasium environment ID
        host: Server host
        port: Server port
    """
    base_url = f"http://{host}:{port}"
    
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    🎮 Gym MCP Server - Running                            ║
╚══════════════════════════════════════════════════════════════════════════╝

📋 Server Information
   └─ Environment: {env_id}
   └─ Version: 0.3.0
   └─ Status: ✓ Running

🌐 API Endpoints
   └─ Base URL: {base_url}
   
   🔧 REST API:
      • POST   {base_url}/reset     - Reset environment to initial state
      • POST   {base_url}/step      - Take an action in the environment
      • POST   {base_url}/render    - Render the current state
      • GET    {base_url}/info      - Get environment information
      • POST   {base_url}/close     - Close the environment
      • GET    {base_url}/health    - Health check endpoint
      • GET    {base_url}/          - Server information
   
   📡 MCP Endpoint:
      • POST   {base_url}/mcp       - MCP (Model Context Protocol) endpoint

📚 Documentation
   └─ Swagger UI:     {base_url}/docs
   └─ OpenAPI Schema: {base_url}/openapi.json
   └─ ReDoc:          {base_url}/redoc

💡 Quick Start
   └─ View API docs:  Open {base_url}/docs in your browser
   └─ Test endpoint:  curl {base_url}/health

═══════════════════════════════════════════════════════════════════════════
"""
    print(banner)


def _generate_index_html(env_id: str, base_url: str) -> str:
    """Generate HTML for the index page.

    Args:
        env_id: The Gymnasium environment ID
        base_url: Base URL of the server

    Returns:
        HTML string
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gym MCP Server - {env_id}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            border-radius: 12px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header .subtitle {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .status-badge {{
            display: inline-block;
            background: #10b981;
            color: white;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 15px;
            font-weight: 500;
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0, 0, 0, 0.15);
        }}
        
        .card h2 {{
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .card h2::before {{
            font-size: 1.2em;
        }}
        
        .endpoint-list {{
            list-style: none;
        }}
        
        .endpoint-item {{
            padding: 12px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .endpoint-item:last-child {{
            border-bottom: none;
        }}
        
        .method {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 0.85em;
            margin-right: 10px;
            min-width: 60px;
            text-align: center;
        }}
        
        .method.post {{
            background: #3b82f6;
            color: white;
        }}
        
        .method.get {{
            background: #10b981;
            color: white;
        }}
        
        .endpoint-path {{
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            color: #667eea;
            font-weight: 500;
        }}
        
        .endpoint-desc {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
            margin-left: 75px;
        }}
        
        .link-button {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            margin: 5px;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .link-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }}
        
        .links-section {{
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        
        .links-section h2 {{
            margin-bottom: 20px;
            color: #667eea;
        }}
        
        .info-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .info-item:last-child {{
            border-bottom: none;
        }}
        
        .info-label {{
            font-weight: 600;
            color: #666;
        }}
        
        .info-value {{
            color: #333;
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
        }}
        
        .footer {{
            text-align: center;
            color: white;
            margin-top: 30px;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 Gym MCP Server</h1>
            <p class="subtitle">Model Context Protocol Server for Gymnasium Environments</p>
            <span class="status-badge">✓ Running</span>
        </div>
        
        <div class="info-grid">
            <div class="card">
                <h2>📋 Server Information</h2>
                <div class="info-item">
                    <span class="info-label">Environment:</span>
                    <span class="info-value">{env_id}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Version:</span>
                    <span class="info-value">0.3.0</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Base URL:</span>
                    <span class="info-value">{base_url}</span>
                </div>
            </div>
            
            <div class="card">
                <h2>🔧 REST API</h2>
                <ul class="endpoint-list">
                    <li class="endpoint-item">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/reset</span>
                        <div class="endpoint-desc">Reset environment to initial state</div>
                    </li>
                    <li class="endpoint-item">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/step</span>
                        <div class="endpoint-desc">Take an action in the environment</div>
                    </li>
                    <li class="endpoint-item">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/render</span>
                        <div class="endpoint-desc">Render the current state</div>
                    </li>
                    <li class="endpoint-item">
                        <span class="method get">GET</span>
                        <span class="endpoint-path">/info</span>
                        <div class="endpoint-desc">Get environment information</div>
                    </li>
                    <li class="endpoint-item">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/close</span>
                        <div class="endpoint-desc">Close the environment</div>
                    </li>
                    <li class="endpoint-item">
                        <span class="method get">GET</span>
                        <span class="endpoint-path">/health</span>
                        <div class="endpoint-desc">Health check endpoint</div>
                    </li>
                </ul>
            </div>
            
            <div class="card">
                <h2>📡 MCP Endpoint</h2>
                <ul class="endpoint-list">
                    <li class="endpoint-item">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/mcp</span>
                        <div class="endpoint-desc">MCP (Model Context Protocol) endpoint</div>
                    </li>
                </ul>
            </div>
        </div>
        
        <div class="links-section">
            <h2>📚 Documentation & Quick Links</h2>
            <a href="/docs" class="link-button">Swagger UI</a>
            <a href="/redoc" class="link-button">ReDoc</a>
            <a href="/openapi.json" class="link-button">OpenAPI Schema</a>
            <a href="/health" class="link-button">Health Check</a>
        </div>
        
        <div class="footer">
            <p>Gym MCP Server v0.3.0 | Powered by FastAPI & Gymnasium</p>
        </div>
    </div>
</body>
</html>"""


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
    )
    
    # Store host and port for banner display
    app.state.host = host
    app.state.port = port
    app.state.env_id = env_id

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

    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request) -> HTMLResponse:
        """Root endpoint providing server information as HTML.

        Returns:
            HTML page with server information and available endpoints
        """
        # Get base URL from request
        base_url = str(request.base_url).rstrip("/")
        html_content = _generate_index_html(env_id=env_id, base_url=base_url)
        return HTMLResponse(content=html_content)

    @app.get("/api", response_model=Dict[str, Any])
    async def api_info() -> Dict[str, Any]:
        """API endpoint providing server information as JSON.

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

    # Add startup event to print banner
    @app.on_event("startup")
    async def startup_event() -> None:
        """Print server banner on startup."""
        if (
            hasattr(app.state, "host")
            and hasattr(app.state, "port")
            and app.state.host is not None
            and app.state.port is not None
        ):
            _print_server_banner(
                env_id=app.state.env_id,
                host=app.state.host,
                port=app.state.port,
            )

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
