"""Phase 2 schemas — TTS configs, audio segments, and timing manifests."""
from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class VoiceConfig(BaseModel):
    """Per-character TTS configuration."""
    character_id: str
    engine: Literal["gtts", "pyttsx3", "elevenlabs", "mock"] = "gtts"
    voice_id: Optional[str] = None
    language: str = "en"
    tld: str = Field(default="com", description="gTTS regional accent: com/co.uk/com.au/...")
    rate: int = Field(default=175, ge=80, le=300, description="words per minute")
    pitch: int = Field(default=0, ge=-50, le=50)
    volume: float = Field(default=1.0, ge=0.0, le=2.0)
    tone: str = Field(default="neutral")


class AudioSegment(BaseModel):
    """A rendered audio clip with timing metadata."""
    segment_id: str
    scene_id: str
    line_id: Optional[str] = None
    character_id: Optional[str] = None
    file_path: str
    kind: Literal["dialogue", "bgm", "sfx"] = "dialogue"
    start_ms: int = Field(..., ge=0)
    end_ms: int = Field(..., ge=0)
    duration_ms: int = Field(..., ge=0)
    text: Optional[str] = None


class TimingManifest(BaseModel):
    """All audio segments with absolute timeline positions."""
    project_id: str
    total_duration_ms: int
    sample_rate: int = 22050
    segments: List[AudioSegment] = Field(default_factory=list)

    def for_scene(self, scene_id: str) -> List[AudioSegment]:
        return [s for s in self.segments if s.scene_id == scene_id]


class AudioOutput(BaseModel):
    """Top-level Phase 2 output."""
    voice_configs: List[VoiceConfig]
    manifest: TimingManifest
    bgm_track: Optional[str] = Field(default=None, description="path to mixed BGM file")
    master_track: Optional[str] = Field(default=None, description="path to mixed master")
