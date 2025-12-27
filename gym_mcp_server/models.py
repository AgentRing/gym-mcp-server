"""Request/Response models for gym environment operations."""

from typing import Any, Optional

from pydantic import BaseModel


class ResetRequest(BaseModel):
    """Request model for reset endpoint."""

    seed: Optional[int] = None


class StepRequest(BaseModel):
    """Request model for step endpoint."""

    action: Any


class RenderRequest(BaseModel):
    """Request model for render endpoint."""

    mode: Optional[str] = None
