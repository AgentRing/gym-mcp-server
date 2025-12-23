"""
Shared service layer for gym environment operations.

This module provides a unified interface for gym operations that can be used
by MCP, HTTP, and gRPC servers to avoid code duplication.
"""

import logging
from typing import Any, Dict, Optional

import gymnasium as gym

from .utils import (
    serialize_observation,
    serialize_render_output,
    serialize_action,
    get_environment_info,
)

logger = logging.getLogger(__name__)


class GymService:
    """Service layer for gym environment operations.

    This class encapsulates all gym operations and provides a unified interface
    that can be used by different protocol implementations (MCP, HTTP, gRPC).
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
    ):
        """Initialize the gym service.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Optional render mode for the environment
        """
        self.env_id = env_id
        self.render_mode = render_mode

        # Initialize the Gymnasium environment
        logger.info(f"Initializing environment: {env_id}")
        if render_mode is not None:
            self.env = gym.make(env_id, render_mode=render_mode)
        else:
            self.env = gym.make(env_id)

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment to its initial state.

        Args:
            seed: Random seed for reproducible episodes (optional)

        Returns:
            Dictionary with observation, info, done, and success status
        """
        logger.info(f"Resetting environment with seed={seed}")
        try:
            if seed is not None:
                obs, info = self.env.reset(seed=seed)
            else:
                obs, info = self.env.reset()

            return {
                "observation": serialize_observation(obs),
                "info": info,
                "done": False,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error resetting environment: {e}")
            return {
                "observation": None,
                "info": {},
                "done": True,
                "success": False,
                "error": str(e),
            }

    def step(self, action: Any) -> Dict[str, Any]:
        """Take an action in the environment.

        Args:
            action: The action to take in the environment

        Returns:
            Dictionary with observation, reward, done, truncated, info,
            and success status
        """
        logger.info(f"Taking step with action={action}")
        try:
            # Convert action to appropriate format if needed
            act: Any = action
            if isinstance(action, (list, tuple)) and len(action) == 1:
                act = action[0]

            obs, reward, done, truncated, info = self.env.step(act)

            return {
                "observation": serialize_observation(obs),
                "reward": float(reward),
                "done": bool(done or truncated),
                "truncated": bool(truncated),
                "info": info,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error taking step: {e}")
            return {
                "observation": None,
                "reward": 0.0,
                "done": True,
                "truncated": False,
                "info": {},
                "success": False,
                "error": str(e),
            }

    def render(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Render the current state of the environment.

        Args:
            mode: Render mode (e.g., rgb_array, human)

        Returns:
            Dictionary with render output, mode, type, shape, and success status
        """
        logger.info(f"Rendering environment with mode={mode}")
        try:
            render_out: Any = self.env.render()
            result = serialize_render_output(render_out, mode or self.render_mode)
            result["success"] = True
            return result
        except Exception as e:
            logger.error(f"Error rendering environment: {e}")
            return {
                "render": None,
                "mode": mode or self.render_mode,
                "type": "error",
                "success": False,
                "error": str(e),
            }

    def get_info(self) -> Dict[str, Any]:
        """Get information about the environment.

        Returns:
            Dictionary with environment metadata and success status
        """
        logger.info("Getting environment info")
        try:
            return {
                "env_info": get_environment_info(self.env),
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error getting environment info: {e}")
            return {
                "env_info": {},
                "success": False,
                "error": str(e),
            }

    def sample(self) -> Dict[str, Any]:
        """Sample a random action from the environment's action space.

        Returns:
            Dictionary with sampled action and success status
        """
        logger.info("Sampling action from action space")
        try:
            action = self.env.action_space.sample()
            return {
                "action": serialize_action(action),
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error sampling action: {e}")
            return {
                "action": None,
                "success": False,
                "error": str(e),
            }

    def close(self) -> Dict[str, Any]:
        """Close the environment and free resources.

        Returns:
            Dictionary with close status and success flag
        """
        logger.info("Closing environment")
        try:
            self.env.close()
            return {"status": "closed", "success": True}
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
            return {"status": "error", "success": False, "error": str(e)}
