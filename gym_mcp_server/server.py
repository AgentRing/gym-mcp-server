"""
MCP server implementation for gym environment operations.

This module provides a GymMCPServer class that wraps GymService and provides
an MCP interface, returning JSON strings for compatibility with MCP clients.
"""

import json
import logging
from typing import Any, Dict, Optional

from .service import GymService

logger = logging.getLogger(__name__)


class GymMCPServer:
    """MCP server wrapper for gym environment operations.

    This class provides an MCP interface to GymService, returning JSON strings
    for compatibility with MCP protocol requirements.
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        **mcp_kwargs: Any,
    ):
        """Initialize the MCP server.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Optional render mode for the environment
            env_kwargs: Optional dictionary of keyword arguments to pass to gym.make()
            **mcp_kwargs: Additional keyword arguments (ignored, kept for compatibility)
        """
        # Extract env_kwargs from mcp_kwargs if it was passed there by mistake
        # This fixes the bug where env_kwargs was being passed to FastMCP
        if "env_kwargs" in mcp_kwargs:
            if env_kwargs is None:
                env_kwargs = mcp_kwargs.pop("env_kwargs")
            else:
                # If both are provided, use the explicit one and remove from mcp_kwargs
                mcp_kwargs.pop("env_kwargs")

        # Initialize the gym service
        self._service = GymService(
            env_id=env_id,
            render_mode=render_mode,
            env_kwargs=env_kwargs,
        )
        self.env_id = env_id
        self.render_mode = render_mode

        # Expose the environment for direct access
        self.env = self._service.env

        # Note: mcp_kwargs are ignored as this class doesn't use FastMCP directly.
        # The MCP functionality is provided by FastApiMCP in the servers module.
        if mcp_kwargs:
            logger.debug(f"Ignoring mcp_kwargs: {list(mcp_kwargs.keys())}")

    def reset_env(self, seed: Optional[int] = None) -> str:
        """Reset the environment to its initial state.

        Args:
            seed: Optional random seed

        Returns:
            JSON string with observation, info, done, success
        """
        result = self._service.reset(seed=seed)
        return json.dumps(result)

    def step_env(self, action: Any) -> str:
        """Take an action in the environment.

        Args:
            action: Action to take

        Returns:
            JSON string with observation, reward, done, truncated, info, success
        """
        result = self._service.step(action)
        return json.dumps(result)

    def render_env(self, mode: Optional[str] = None) -> str:
        """Render the current state of the environment.

        Args:
            mode: Optional render mode

        Returns:
            JSON string with render output, mode, type, shape, success
        """
        result = self._service.render(mode=mode)
        return json.dumps(result)

    def get_env_info(self) -> str:
        """Get information about the environment.

        Returns:
            JSON string with environment metadata, success
        """
        result = self._service.get_info()
        return json.dumps(result)

    def close_env(self) -> str:
        """Close the environment and free resources.

        Returns:
            JSON string with close status, success
        """
        result = self._service.close()
        return json.dumps(result)
