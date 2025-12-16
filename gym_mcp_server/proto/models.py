"""
Pydantic models generated from Protocol Buffer definitions.

This file is auto-generated from gym_service.proto.
DO NOT EDIT MANUALLY. To regenerate, run:
    python scripts/generate_pydantic_models.py

The models are derived from:
    gym_mcp_server/proto/gym_service.proto

These models are adapted for HTTP/FastAPI use, using simpler types
(Any for observations/actions) compared to the structured proto types.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResetRequest(BaseModel):
    """Request model for resetting the environment (generated from proto)."""

    seed: Optional[int] = Field(
        default=None, description="Random seed for reproducible episodes"
    )


class ResetResponse(BaseModel):
    """Response model for environment reset (generated from proto)."""

    observation: Any = Field(description="Initial observation from the environment")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional information from the environment"
    )
    done: bool = Field(
        default=False, description="Whether the episode is done (always False for reset)"
    )
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class StepRequest(BaseModel):
    """Request model for taking a step in the environment (generated from proto)."""

    action: Any = Field(description="Action to take in the environment")


class StepResponse(BaseModel):
    """Response model for environment step (generated from proto)."""

    observation: Any = Field(description="Next observation from the environment")
    reward: float = Field(description="Reward received from the action")
    done: bool = Field(description="Whether the episode is done")
    truncated: bool = Field(default=False, description="Whether the episode was truncated")
    info: Dict[str, Any] = Field(
        default_factory=dict, description="Additional information from the environment"
    )
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class RenderRequest(BaseModel):
    """Request model for rendering the environment (generated from proto)."""

    mode: Optional[str] = Field(
        default=None, description="Render mode (e.g., rgb_array, human)"
    )


class RenderResponse(BaseModel):
    """Response model for environment render (generated from proto)."""

    render: Any = Field(description="Rendered output from the environment")
    mode: Optional[str] = Field(description="Render mode used")
    type: Optional[str] = Field(
        default=None,
        description="Type of render output (text, image_base64, array, string)",
    )
    shape: Optional[List[int]] = Field(
        default=None, description="Shape of the render output (for arrays/images)"
    )
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class GetInfoRequest(BaseModel):
    """Request model for getting environment info (generated from proto)."""

    pass


class GetInfoResponse(BaseModel):
    """Response model for environment info (generated from proto)."""

    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    env_info: Dict[str, Any] = Field(description="Environment metadata")


class EnvInfoResponse(BaseModel):
    """Response model for environment info (HTTP API wrapper)."""

    env_info: Dict[str, Any] = Field(description="Environment metadata")
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class CloseRequest(BaseModel):
    """Request model for closing the environment (generated from proto)."""

    pass


class CloseResponse(BaseModel):
    """Response model for closing the environment (generated from proto)."""

    status: str = Field(description="Status of the close operation")
    success: bool = Field(description="Whether the operation was successful")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthRequest(BaseModel):
    """Request model for health check (generated from proto)."""

    pass


class HealthResponse(BaseModel):
    """Response model for health check (generated from proto)."""

    status: str = Field(description="Server status")
    env_id: str = Field(description="Currently loaded environment ID")
    render_mode: Optional[str] = Field(description="Current render mode")

