"""
MCP server implementation for exposing Gymnasium environments as MCP tools.
"""

import json
import logging
import threading
import time
from typing import Any, Literal, Optional

from mcp.server.fastmcp import FastMCP

from .base import BaseGymServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GymMCPServer(BaseGymServer):
    """MCP Server for Gymnasium environments.

    This class manages a FastMCP server instance and exposes Gymnasium
    environment operations as MCP tools.

    Example (blocking):
        >>> server = GymMCPServer(env_id="CartPole-v1")
        >>> server.run(transport="stdio")

    Example (non-blocking):
        >>> server = GymMCPServer(env_id="CartPole-v1", host="localhost", port=8765)
        >>> server.start(transport="streamable-http")  # Start in background thread
        >>> # ... do other work ...
        >>> server.stop()  # Stop the server
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        **mcp_kwargs: Any,
    ):
        """Initialize the Gym MCP Server.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Optional render mode for the environment
            **mcp_kwargs: Additional FastMCP settings like host, port, etc.
        """
        super().__init__(env_id=env_id, render_mode=render_mode)
        self._mcp_kwargs = mcp_kwargs

        # Create the FastMCP server (private)
        self._mcp = FastMCP("gym-mcp-server", **mcp_kwargs)

        # MCP-specific stop event
        self._stop_event = threading.Event()

        # Register all tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all environment tools with the MCP server."""
        self._mcp.tool()(self.reset_env)
        self._mcp.tool()(self.step_env)
        self._mcp.tool()(self.render_env)
        self._mcp.tool()(self.get_env_info)
        self._mcp.tool()(self.close_env)

    def run(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
    ) -> None:
        """Run the MCP server (blocking).

        Args:
            transport: Transport type (e.g., "stdio", "streamable-http", "sse")
        """
        self._mcp.run(transport=transport)

    def _run_in_thread(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http",
    ) -> None:
        """Internal method to run server in a thread."""
        host = self._mcp_kwargs.get("host", "localhost")
        port = self._mcp_kwargs.get("port", 8765)
        logger.info(f"Starting MCP server on {host}:{port}")

        try:
            self._mcp.run(transport=transport)
        except Exception as e:
            if not self._stop_event.is_set():
                logger.error(f"Server error: {e}")
            else:
                logger.info("Server stopped gracefully")

    def start(
        self,
        transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http",
    ) -> None:
        """Start the server in a background thread (non-blocking).

        Args:
            transport: Transport type (default: "streamable-http")

        Raises:
            RuntimeError: If server is already running

        Example:
            >>> server = GymMCPServer(env_id="CartPole-v1", host="localhost", port=8765)
            >>> server.start(transport="streamable-http")
            >>> # Server is now running at http://localhost:8765/mcp
            >>> server.stop()
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Server already started")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_in_thread, args=(transport,), daemon=True
        )
        self._thread.start()

        # Wait a bit for the server to start up
        logger.info("Waiting for server to start...")
        time.sleep(2)

        host = self._mcp_kwargs.get("host", "localhost")
        port = self._mcp_kwargs.get("port", 8765)
        logger.info(f"Server should be ready at http://{host}:{port}")

    def stop(self) -> None:
        """Stop the server thread if running in non-blocking mode.

        Note: FastMCP servers don't have a clean shutdown mechanism.
        This method sets a stop event and relies on the daemon thread
        to terminate with the main process.
        """
        if self._thread is None:
            return

        logger.info("Stopping server...")
        self._stop_event.set()

    def reset_env(self, seed: Optional[int] = None) -> str:
        """Reset the environment to its initial state.

        Args:
            seed: Random seed for reproducible episodes (optional)

        Returns:
            JSON string with initial observation, info, and done status
        """
        result = self._service.reset(seed=seed)
        return json.dumps(result, indent=2)

    def step_env(self, action: Any) -> str:
        """Take an action in the environment.

        Args:
            action: The action to take in the environment

        Returns:
            JSON string with next observation, reward, done status, and info
        """
        result = self._service.step(action)
        return json.dumps(result, indent=2)

    def render_env(self, mode: Optional[str] = None) -> str:
        """Render the current state of the environment.

        Args:
            mode: Render mode (e.g., rgb_array, human)

        Returns:
            JSON string with rendered output
        """
        result = self._service.render(mode)
        return json.dumps(result, indent=2)

    def get_env_info(self) -> str:
        """Get information about the environment.

        Returns:
            JSON string with environment metadata
        """
        result = self._service.get_info()
        return json.dumps(result, indent=2)

    def close_env(self) -> str:
        """Close the environment and free resources.

        Returns:
            JSON string with close status
        """
        result = self._service.close()
        return json.dumps(result, indent=2)

