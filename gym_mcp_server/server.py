"""
Main MCP server implementation for exposing Gymnasium environments as MCP tools.
"""

import gymnasium as gym
import json
import logging
import threading
import time
from typing import Any, Literal, Optional
from mcp.server.fastmcp import FastMCP
from .utils import (
    serialize_observation,
    serialize_render_output,
    get_environment_info,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GymMCPServer:
    """MCP Server for Gymnasium environments.

    This class manages a FastMCP server instance and exposes Gymnasium
    environment operations as MCP tools.

    Example (blocking):
        >>> server = GymMCPServer(env_id="CartPole-v1")
        >>> server.run(transport="stdio")

    Example (non-blocking):
        >>> server = GymMCPServer(env_id="CartPole-v1", host="localhost", port=8765)
        >>> server.start()  # Start in background thread
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
        self.env_id = env_id
        self.render_mode = render_mode
        self._mcp_kwargs = mcp_kwargs

        # Create the FastMCP server (private)
        self._mcp = FastMCP("gym-mcp-server", **mcp_kwargs)

        # Initialize the Gymnasium environment
        logger.info(f"Initializing environment: {env_id}")
        if render_mode is not None:
            self.env = gym.make(env_id, render_mode=render_mode)
        else:
            self.env = gym.make(env_id)

        # Threading support for non-blocking mode
        self._thread: Optional[threading.Thread] = None
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

        This allows running the server while continuing to execute other code
        in the main thread. Useful for testing, embedded applications, or when
        you need to run server and client in the same Python process.

        Args:
            transport: Transport type (default: "streamable-http")

        Raises:
            RuntimeError: If server is already running

        Example:
            >>> server = GymMCPServer(env_id="CartPole-v1", host="localhost", port=8765)
            >>> server.start()
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

    def is_alive(self) -> bool:
        """Check if the server thread is alive.

        Returns:
            True if the server thread is running, False otherwise
        """
        return self._thread is not None and self._thread.is_alive()

    def reset_env(self, seed: Optional[int] = None) -> str:
        """Reset the environment to its initial state.

        Args:
            seed: Random seed for reproducible episodes (optional)

        Returns:
            JSON string with initial observation, info, and done status
        """
        logger.info(f"Resetting environment with seed={seed}")
        try:
            if seed is not None:
                obs, info = self.env.reset(seed=seed)
            else:
                obs, info = self.env.reset()

            result = {
                "observation": serialize_observation(obs),
                "info": info,
                "done": False,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            result = {
                "observation": None,
                "info": {},
                "done": True,
                "success": False,
                "error": str(e),
            }
        return json.dumps(result, indent=2)

    def step_env(self, action: Any) -> str:
        """Take an action in the environment.

        Args:
            action: The action to take in the environment

        Returns:
            JSON string with next observation, reward, done status, and info
        """
        logger.info(f"Taking step with action={action}")
        try:
            # Convert action to appropriate format if needed
            act: Any = action
            if isinstance(action, (list, tuple)) and len(action) == 1:
                act = action[0]

            obs, reward, done, truncated, info = self.env.step(act)

            result = {
                "observation": serialize_observation(obs),
                "reward": float(reward),
                "done": bool(done or truncated),
                "truncated": bool(truncated),
                "info": info,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error taking step: {e}")
            result = {
                "observation": None,
                "reward": 0.0,
                "done": True,
                "truncated": False,
                "info": {},
                "success": False,
                "error": str(e),
            }
        return json.dumps(result, indent=2)

    def render_env(self, mode: Optional[str] = None) -> str:
        """Render the current state of the environment.

        Args:
            mode: Render mode (e.g., rgb_array, human)

        Returns:
            JSON string with rendered output
        """
        logger.info(f"Rendering environment with mode={mode}")
        try:
            render_out: Any = self.env.render()
            result = serialize_render_output(render_out, mode or self.render_mode)
            result["success"] = True
        except Exception as e:
            logger.error(f"Error rendering environment: {e}")
            result = {
                "render": None,
                "mode": mode or self.render_mode,
                "type": "error",
                "success": False,
                "error": str(e),
            }
        return json.dumps(result, indent=2)

    def get_env_info(self) -> str:
        """Get information about the environment.

        Returns:
            JSON string with environment metadata
        """
        logger.info("Getting environment info")
        try:
            result = {"env_info": get_environment_info(self.env), "success": True}
        except Exception as e:
            logger.error(f"Error getting environment info: {e}")
            result = {"env_info": {}, "success": False, "error": str(e)}
        return json.dumps(result, indent=2)

    def close_env(self) -> str:
        """Close the environment and free resources.

        Returns:
            JSON string with close status
        """
        logger.info("Closing environment")
        try:
            self.env.close()
            result = {"status": "closed", "success": True}
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
            result = {"status": "error", "success": False, "error": str(e)}
        return json.dumps(result, indent=2)
