"""Phase 3 schemas — visual prompts, scene frames, and final composition."""
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class VisualPrompt(BaseModel):
    """Image-generation request for a single scene."""
    scene_id: str
    prompt: str
    negative_prompt: str = ""
    style: str = Field(default="cinematic, detailed, dramatic lighting")
    aspect_ratio: str = "16:9"
    seed: Optional[int] = None


class CharacterPortrait(BaseModel):
    """A close-up portrait used as a talking-head shot."""
    character_id: str
    image_path: str
    is_lip_synced: bool = False
    talking_head_clip: Optional[str] = None  # if provider produced a real clip


class Shot(BaseModel):
    """One sub-clip inside a scene (a single 'cut')."""
    shot_id: str
    scene_id: str
    kind: Literal["establishing", "character", "lip_sync", "broll"] = "establishing"
    character_id: Optional[str] = None
    image_path: str
    clip_path: Optional[str] = None
    duration_ms: int
    motion: str = "ken_burns_diag"
    audio_path: Optional[str] = None     # the dialogue line whose duration drives this shot


class SceneFrame(BaseModel):
    """A rendered scene with one or more shots and optional final composite."""
    scene_id: str
    image_path: str                       # establishing image (kept for back-compat)
    clip_path: Optional[str] = None       # final per-scene composite (multi-shot if dialogue, else single)
    width: int = 1280
    height: int = 720
    duration_ms: int
    motion: str = "ken_burns_diag"
    transition_in: str = "fade"
    shots: List[Shot] = Field(default_factory=list)


class VideoOutput(BaseModel):
    """Top-level Phase 3 output."""
    project_id: str
    frames: List[SceneFrame]
    final_video_path: str
    width: int = 1280
    height: int = 720
    fps: int = 24
    has_subtitles: bool = False
    duration_ms: int = 0
    portraits: List[CharacterPortrait] = Field(default_factory=list)
    used_text_to_video: bool = False
    used_lip_sync: bool = False
