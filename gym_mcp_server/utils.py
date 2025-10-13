"""
Utility functions for serialization, rendering, and other helper operations.
"""

import base64
import io
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image


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
