"""Phase 5 schemas — edit intent classification, commands, and results."""
from __future__ import annotations
from typing import Optional, Dict, Any, Literal, List
from pydantic import BaseModel, Field


EditTarget = Literal["audio", "video_frame", "video", "script"]


class EditIntent(BaseModel):
    """Structured output of the LLM intent classifier."""
    intent: str = Field(..., description="e.g. change_voice_tone, regenerate_scene")
    target: EditTarget
    scope: str = Field(
        default="global",
        description="e.g. character:Narrator / scene:scene_2 / global",
    )
    parameters: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class EditCommand(BaseModel):
    """A user-issued natural-language edit."""
    project_id: str
    query: str
    user_id: Optional[str] = None


class EditResult(BaseModel):
    """Result of executing an edit."""
    success: bool
    intent: EditIntent
    new_version: Optional[int] = None
    affected_assets: List[str] = Field(default_factory=list)
    message: str = ""
    error: Optional[str] = None
