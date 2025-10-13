"""
JSON schema definitions for observations, actions, and other data structures.
"""

# Schema for environment reset response
RESET_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"description": "Initial observation from the environment"},
        "info": {
            "type": "object",
            "description": "Additional information from the environment",
        },
        "done": {
            "type": "boolean",
            "description": "Whether the episode is done (always False for reset)",
        },
    },
    "required": ["observation", "info", "done"],
}

# Schema for environment step response
STEP_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"description": "Next observation from the environment"},
        "reward": {"type": "number", "description": "Reward received from the action"},
        "done": {"type": "boolean", "description": "Whether the episode is done"},
        "info": {
            "type": "object",
            "description": "Additional information from the environment",
        },
    },
    "required": ["observation", "reward", "done", "info"],
}

# Schema for render response
RENDER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "render": {"description": "Rendered output from the environment"},
        "mode": {"type": "string", "description": "Render mode used"},
        "type": {
            "type": "string",
            "description": "Type of render output (text, image_base64, array, string)",
        },
        "shape": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Shape of the render output (for arrays/images)",
        },
    },
    "required": ["render", "mode"],
}

# Schema for environment info response
ENV_INFO_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Environment ID"},
        "action_space": {
            "type": "string",
            "description": "String representation of action space",
        },
        "observation_space": {
            "type": "string",
            "description": "String representation of observation space",
        },
        "reward_range": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Range of possible rewards",
        },
        "action_space_size": {
            "type": "integer",
            "description": "Size of discrete action space",
        },
        "action_space_shape": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Shape of continuous action space",
        },
        "observation_space_shape": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Shape of observation space",
        },
        "observation_space_spaces": {
            "type": "object",
            "description": "Spaces for dictionary observation spaces",
        },
    },
}

# Schema for close response
CLOSE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "description": "Status of the close operation"}
    },
    "required": ["status"],
}

# Tool schemas for MCP
TOOL_SCHEMAS = {
    "reset_env": {
        "name": "reset_env",
        "description": "Reset the environment to its initial state",
        "inputSchema": {
            "type": "object",
            "properties": {
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducible episodes (optional)",
                }
            },
        },
    },
    "step_env": {
        "name": "step_env",
        "description": (
            "Take an action in the environment and get the next "
            "observation, reward, and done status"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"description": "Action to take in the environment"}
            },
            "required": ["action"],
        },
    },
    "render_env": {
        "name": "render_env",
        "description": "Render the current state of the environment",
        "inputSchema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "description": "Render mode (e.g., rgb_array, human)",
                }
            },
        },
    },
    "close_env": {
        "name": "close_env",
        "description": "Close the environment and free resources",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "get_env_info": {
        "name": "get_env_info",
        "description": (
            "Get information about the environment "
            "(action space, observation space, etc.)"
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
}
