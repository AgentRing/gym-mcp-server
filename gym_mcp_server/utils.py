"""
Utility functions for serialization, rendering, and other helper operations.
"""

import base64
import hashlib
import importlib
import io
import logging
from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def serialize_observation(obs: Any) -> Any:
    """
    Convert Gymnasium observations to JSON-safe formats.

    Args:
        obs: The observation from the environment

    Returns:
        JSON-serializable representation of the observation
    """
    # Handle numpy types first
    if isinstance(obs, np.integer):
        return int(obs)
    elif isinstance(obs, np.floating):
        return float(obs)
    elif isinstance(obs, np.bool_):
        return bool(obs)
    elif isinstance(obs, np.ndarray):
        return obs.tolist()

    # Handle basic Python types
    if isinstance(obs, (int, float, str, bool, type(None))):
        return obs

    # Handle numpy arrays
    if hasattr(obs, "tolist"):
        return obs.tolist()

    # Handle numpy scalars
    if hasattr(obs, "item"):
        return obs.item()

    # Handle dictionaries
    if isinstance(obs, dict):
        return {k: serialize_observation(v) for k, v in obs.items()}

    # Handle lists/tuples
    if isinstance(obs, (list, tuple)):
        return [serialize_observation(item) for item in obs]

    # Fallback to string representation
    return str(obs)


def serialize_action(action: Any) -> Any:
    """
    Convert action to JSON-safe format.

    Args:
        action: The action to serialize

    Returns:
        JSON-serializable representation of the action
    """
    if isinstance(action, (int, float, str, bool, type(None))):
        return action

    # Handle numpy arrays
    if hasattr(action, "tolist"):
        return action.tolist()

    # Handle numpy scalars
    if hasattr(action, "item"):
        return action.item()

    # Handle dictionaries
    if isinstance(action, dict):
        return {k: serialize_action(v) for k, v in action.items()}

    # Handle lists/tuples
    if isinstance(action, (list, tuple)):
        return [serialize_action(item) for item in action]

    # Handle numpy arrays
    if isinstance(action, np.ndarray):
        return action.tolist()

    return action


def serialize_render_output(
    render_out: Any, mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Serialize render output to JSON-safe format.

    Args:
        render_out: The render output from env.render()
        mode: The render mode used (e.g., rgb_array, human)

    Returns:
        Dictionary containing the serialized render output
    """
    if render_out is None:
        return {"render": None, "mode": mode}

    # Handle RGB arrays (images)
    if isinstance(render_out, np.ndarray) and len(render_out.shape) == 3:
        # Convert RGB array to base64 for JSON transport
        if render_out.dtype == np.uint8:
            # Encode as base64
            img = Image.fromarray(render_out)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return {
                "render": img_str,
                "mode": mode,
                "type": "image_base64",
                "shape": render_out.shape,
            }
        else:
            # Convert to list for JSON serialization
            return {
                "render": render_out.tolist(),
                "mode": mode,
                "type": "array",
                "shape": render_out.shape,
            }

    # Handle text output
    if isinstance(render_out, str):
        return {"render": render_out, "mode": mode, "type": "text"}

    # Handle numpy arrays
    if hasattr(render_out, "tolist"):
        return {"render": render_out.tolist(), "mode": mode, "type": "array"}

    # Fallback to string representation
    return {"render": str(render_out), "mode": mode, "type": "string"}


def get_environment_info(env: Any) -> Dict[str, Any]:
    """
    Extract useful information about the environment.

    Args:
        env: The Gymnasium environment

    Returns:
        Dictionary containing environment metadata
    """
    info = {
        "id": (
            getattr(env.spec, "id", None) if hasattr(env, "spec") and env.spec else None
        ),
        "action_space": str(env.action_space),
        "observation_space": str(env.observation_space),
        "reward_range": getattr(env, "reward_range", None),
    }

    # Add action space details
    if hasattr(env.action_space, "n"):
        try:
            info["action_space_size"] = int(
                env.action_space.n
            )  # Convert numpy int to Python int
        except (TypeError, ValueError):
            pass  # Skip if conversion fails (e.g., Mock objects in tests)
    elif hasattr(env.action_space, "shape"):
        try:
            info["action_space_shape"] = list(
                env.action_space.shape
            )  # Convert numpy array to list
        except (TypeError, ValueError):
            pass  # Skip if conversion fails (e.g., Mock objects in tests)

    # Add observation space details
    if hasattr(env.observation_space, "shape"):
        try:
            info["observation_space_shape"] = list(
                env.observation_space.shape
            )  # Convert numpy array to list
        except (TypeError, ValueError):
            pass  # Skip if conversion fails (e.g., Mock objects in tests)
    elif hasattr(env.observation_space, "spaces"):
        try:
            info["observation_space_spaces"] = {
                k: str(v) for k, v in env.observation_space.spaces.items()
            }
        except (TypeError, ValueError, AttributeError):
            pass  # Skip if conversion fails (e.g., Mock objects in tests)

    return info


def parse_entry_point(entry_point: str) -> Tuple[str, str]:
    """
    Parse an entry point string into module path and class name.

    Args:
        entry_point: Entry point in format "module.path:ClassName"

    Returns:
        Tuple of (module_path, class_name)

    Raises:
        ValueError: If entry point format is invalid
    """
    if ":" not in entry_point:
        raise ValueError(
            f"Invalid entry point format: '{entry_point}'. "
            "Expected format: 'module.path:ClassName'"
        )

    parts = entry_point.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid entry point format: '{entry_point}'. "
            "Expected format: 'module.path:ClassName'"
        )

    return parts[0], parts[1]


def register_env_from_entry_point(
    entry_point: str,
    env_id: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Register a Gymnasium environment from a module:class entry point.

    This allows using any Python class that implements the Gymnasium Env interface
    as an environment without pre-registering it.

    Args:
        entry_point: Entry point in format "module.path:ClassName"
        env_id: Optional environment ID. If not provided, one will be generated.
        env_kwargs: Optional kwargs to pass to the environment constructor.

    Returns:
        The registered environment ID

    Raises:
        ValueError: If entry point format is invalid
        ImportError: If the module cannot be imported
        AttributeError: If the class is not found in the module

    Example:
        >>> env_id = register_env_from_entry_point(
        ...     "mymodule.envs:MyCustomEnv",
        ...     env_kwargs={"difficulty": "hard"}
        ... )
        >>> env = gym.make(env_id)
    """
    module_path, class_name = parse_entry_point(entry_point)

    # Validate that the module and class exist
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}' "
            f"from entry point '{entry_point}': {e}"
        ) from e

    if not hasattr(module, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_path}'"
        )

    # Generate an environment ID if not provided
    if env_id is None:
        # Create a unique ID based on the entry point
        hash_suffix = hashlib.md5(entry_point.encode()).hexdigest()[:8]
        env_id = f"{class_name}-dynamic-{hash_suffix}-v0"

    # Check if already registered
    if env_id in gym.envs.registry:
        logger.info(f"Environment '{env_id}' already registered, skipping registration")
        return env_id

    # Register the environment
    logger.info(f"Registering environment '{env_id}' from entry point '{entry_point}'")
    register(
        id=env_id,
        entry_point=entry_point,
        kwargs=env_kwargs or {},
    )

    return env_id
