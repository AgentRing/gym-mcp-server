"""
Shared service layer for gym environment operations.

This module provides a unified interface for gym operations that can be used
by MCP and HTTP servers to avoid code duplication.
"""

import logging
from typing import Any, Dict, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .utils import (
    serialize_observation,
    serialize_render_output,
    serialize_action,
    get_environment_info,
)

logger = logging.getLogger(__name__)


def _auto_detect_render_mode(env_id: str) -> Optional[str]:
    """Try to automatically detect and enable a suitable render mode.
    
    Args:
        env_id: The Gymnasium environment ID
        
    Returns:
        A suitable render mode if available, None otherwise
    """
    # Common render modes to try in order of preference
    render_modes_to_try = ["rgb_array", "ansi", "human"]
    
    for mode in render_modes_to_try:
        try:
            # Try to create the environment with this render mode
            test_env = gym.make(env_id, render_mode=mode)
            test_env.close()
            return mode
        except (ValueError, TypeError, AttributeError):
            # This render mode is not supported, try the next one
            continue
        except Exception:
            # Some other error occurred, skip this mode
            continue
    
    # No suitable render mode found
    return None


class GymService:
    """Service layer for gym environment operations.

    This class encapsulates all gym operations and provides a unified interface
    that can be used by different protocol implementations (MCP, HTTP).
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
    ):
        """Initialize the gym service.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Optional render mode for the environment.
                        If None, will attempt to auto-enable a suitable render mode.
        """
        self.env_id = env_id
        self._has_been_reset = False

        # Initialize the Gymnasium environment
        logger.info(f"Initializing environment: {env_id}")
        
        # If render_mode is not provided, try to auto-enable rendering
        if render_mode is None:
            render_mode = _auto_detect_render_mode(env_id)
            if render_mode:
                logger.info(f"Auto-enabled render mode: {render_mode}")
        
        self.render_mode = render_mode
        if render_mode is not None:
            self.env = gym.make(env_id, render_mode=render_mode)
        else:
            self.env = gym.make(env_id)
        logger.info(f"Successfully initialized environment: {env_id}")

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

            self._has_been_reset = True
            return {
                "observation": serialize_observation(obs),
                "info": serialize_observation(info),
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

            # Check if action space is text-based (e.g., TextWorld)
            # Only convert string actions to integers for discrete action spaces
            is_text_space = isinstance(self.env.action_space, spaces.Text)

            # Convert string actions to integers only for discrete action spaces
            if isinstance(act, str) and not is_text_space:
                try:
                    act = int(act)
                    logger.debug(f"Converted string action '{action}' to integer {act}")
                except (ValueError, TypeError):
                    # If conversion fails, try to extract number from string
                    import re

                    match = re.search(r"\d+", str(act))
                    if match:
                        act = int(match.group())
                        logger.debug(
                            f"Extracted and converted action '{action}' "
                            f"to integer {act}"
                        )
                    else:
                        raise ValueError(f"Cannot convert action '{act}' to integer")

            # Validate action against action space
            action_space = self.env.action_space

            # For discrete action spaces, check if action is in valid range
            if isinstance(action_space, spaces.Discrete):
                if not isinstance(act, (int, np.integer)):
                    raise ValueError(
                        f"Invalid action type for Discrete action space: "
                        f"expected int, got {type(act).__name__}"
                    )
                act = int(act)
                if act < 0 or act >= action_space.n:
                    raise ValueError(
                        f"Action {act} is out of bounds for Discrete action space "
                        f"with size {action_space.n}. "
                        f"Valid actions are in range [0, {action_space.n - 1}]"
                    )

            # For Box action spaces, check bounds
            elif isinstance(action_space, spaces.Box):
                if isinstance(act, (int, float, np.number)):
                    act = np.array([act])
                elif isinstance(act, (list, tuple)):
                    act = np.array(act)
                elif not isinstance(act, np.ndarray):
                    raise ValueError(
                        f"Invalid action type for Box action space: "
                        f"expected array-like, got {type(act).__name__}"
                    )

                act = np.asarray(act)
                if act.shape != action_space.shape:
                    raise ValueError(
                        f"Action shape {act.shape} does not match action space shape "
                        f"{action_space.shape}"
                    )

                # Check if action is within bounds
                if not action_space.contains(act):
                    low_str = str(action_space.low)
                    high_str = str(action_space.high)
                    raise ValueError(
                        f"Action {act} is out of bounds for Box action space. "
                        f"Bounds: low={low_str}, high={high_str}"
                    )

            # For MultiDiscrete action spaces
            elif isinstance(action_space, spaces.MultiDiscrete):
                if isinstance(act, (int, float, np.number)):
                    act = np.array([act])
                elif isinstance(act, (list, tuple)):
                    act = np.array(act)
                elif not isinstance(act, np.ndarray):
                    raise ValueError(
                        f"Invalid action type for MultiDiscrete action space: "
                        f"expected array-like, got {type(act).__name__}"
                    )

                act = np.asarray(act, dtype=np.int32)
                if act.shape != action_space.shape:
                    raise ValueError(
                        f"Action shape {act.shape} does not match action space shape "
                        f"{action_space.shape}"
                    )

                # Check bounds for each dimension
                for i, (a, n) in enumerate(zip(act.flat, action_space.nvec.flat)):
                    if a < 0 or a >= n:
                        raise ValueError(
                            f"Action dimension {i} value {a} is out of bounds. "
                            f"Valid range is [0, {n - 1}]"
                        )

            obs, reward, done, truncated, info = self.env.step(act)

            return {
                "observation": serialize_observation(obs),
                "reward": float(reward),
                "done": bool(done or truncated),
                "terminated": bool(done),
                "truncated": bool(truncated),
                "info": serialize_observation(info),
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
            mode: Render mode (e.g., rgb_array, human). If None, uses the mode
                  set during initialization.

        Returns:
            Dictionary with render output, mode, type, shape, and success status
        """
        # Use provided mode, or fall back to the render_mode set during init
        render_mode = mode or self.render_mode
        logger.info(f"Rendering environment with mode={render_mode}")
        try:
            # Ensure environment has been reset before rendering
            if not self._has_been_reset:
                logger.info("Environment not reset yet, auto-resetting before render")
                self.env.reset()
                self._has_been_reset = True

            # If environment was created with a render_mode, it will use that automatically
            # when render() is called without arguments
            render_out: Any = self.env.render()
            
            result = serialize_render_output(render_out, render_mode)
            result["success"] = True
            return result
        except Exception as e:
            logger.error(f"Error rendering environment: {e}")
            return {
                "render": None,
                "mode": render_mode,
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
