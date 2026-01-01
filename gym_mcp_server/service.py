"""
Shared service layer for gym environment operations.

This module provides a unified interface for gym operations that can be used
by MCP, HTTP, and gRPC servers to avoid code duplication.
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


class GymService:
    """Service layer for gym environment operations.

    This class encapsulates all gym operations and provides a unified interface
    that can be used by different protocol implementations (MCP, HTTP, gRPC).
    """

    def __init__(
        self,
        env_id: str,
        render_mode: Optional[str] = None,
        enable_run_manager: bool = True,
        num_episodes: int = 10,
        max_steps_per_episode: int = 1000,
    ):
        """Initialize the gym service.

        Args:
            env_id: The Gymnasium environment ID
            render_mode: Optional render mode for the environment
            enable_run_manager: Whether to enable run manager for lifecycle tracking
            num_episodes: Number of episodes in a run (used if run manager enabled)
            max_steps_per_episode: Maximum steps per episode
                (used if run manager enabled)
        """
        self.env_id = env_id
        self.render_mode = render_mode

        # Initialize the Gymnasium environment
        logger.info(f"Initializing environment: {env_id}")
        if render_mode is not None:
            self.env = gym.make(env_id, render_mode=render_mode)
        else:
            self.env = gym.make(env_id)
        logger.info(f"Successfully initialized environment: {env_id}")

        # Initialize run manager if enabled
        self.run_manager: Optional["BasicRunManager"] = None
        if enable_run_manager:
            from .run_manager import BasicRunManager

            self.run_manager = BasicRunManager(
                gym_service=self,
                num_episodes=num_episodes,
                max_steps_per_episode=max_steps_per_episode,
            )
            # Don't auto-start - wait for explicit start_run() call from client

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

            result = {
                "observation": serialize_observation(obs),
                "info": info,
                "done": False,
                "success": True,
            }

            # Notify run manager of episode start
            if self.run_manager:
                try:
                    episode_info = self.run_manager.start_episode(seed=seed)
                    result["run_progress"] = {
                        "episode_num": episode_info["episode_num"],
                        "total_episodes": episode_info["total_episodes"],
                        "max_steps": episode_info["max_steps"],
                    }
                except RuntimeError as e:
                    # If run manager is enabled, enforce that start_run() was called
                    error_msg = str(e)
                    if "Cannot start episode" in error_msg:
                        logger.error(
                            f"Run manager error: {error_msg}. "
                            "Call start_run() before reset() to begin "
                            "tracking episodes."
                        )
                        return {
                            "observation": None,
                            "info": {},
                            "done": True,
                            "success": False,
                            "error": (
                                f"Run not started. Call start_run() first: "
                                f"{error_msg}"
                            ),
                        }
                    else:
                        logger.warning(f"Run manager error: {e}")

            return result
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

            result = {
                "observation": serialize_observation(obs),
                "reward": float(reward),
                "done": bool(done or truncated),
                "terminated": bool(done),
                "truncated": bool(truncated),
                "info": info,
                "success": True,
            }

            # Record step in run manager
            if self.run_manager:
                try:
                    step_info = self.run_manager.record_step(
                        action=action,
                        observation=result.get("observation"),
                        reward=result.get("reward", 0.0),
                        done=result.get("done", False),
                        truncated=result.get("truncated", False),
                        info=result.get("info", {}),
                    )
                    # Add run progress to result
                    if step_info:
                        result["run_progress"] = step_info.get("run_progress", {})
                except RuntimeError as e:
                    # If run manager is enabled, enforce that start_run() was called
                    error_msg = str(e)
                    if (
                        "Cannot record step" in error_msg
                        or "no active episode" in error_msg
                    ):
                        logger.warning(
                            f"Run manager error: {error_msg}. "
                            "Call start_run() and reset() before step() "
                            "to begin tracking."
                        )
                        # Don't fail the step, just log the warning
                        # The step itself succeeded, we just couldn't track it
                    else:
                        logger.warning(f"Run manager error: {e}")
                except AttributeError as e:
                    # Handle case where current_episode_stats is None
                    error_msg = str(e)
                    if "total_steps" in error_msg or "no attribute" in error_msg:
                        logger.warning(
                            f"Run manager error: {error_msg}. "
                            "No active episode to record step. "
                            "This may happen if an episode completed and "
                            "a new one hasn't been started yet."
                        )
                        # Don't fail the step, just log the warning
                    else:
                        logger.warning(f"Run manager error: {e}")

            return result
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
