"""Audio tools — TTS, BGM, audio merging."""
from .tts_tool import TtsTool
from .bgm_tool import BgmTool
from .audio_merger import AudioMergerTool
from mcp.tool_registry import registry

registry.register(TtsTool())
registry.register(BgmTool())
registry.register(AudioMergerTool())

__all__ = ["TtsTool", "BgmTool", "AudioMergerTool"]
