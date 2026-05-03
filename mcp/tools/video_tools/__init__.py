"""Video tools — ffmpeg ops, scene compositor, subtitle overlay."""
from .ffmpeg_tool import FfmpegTool, ImageToClipTool
from .compositor_tool import CompositorTool
from .subtitle_tool import SubtitleTool
from mcp.tool_registry import registry

registry.register(FfmpegTool())
registry.register(ImageToClipTool())
registry.register(CompositorTool())
registry.register(SubtitleTool())

__all__ = ["FfmpegTool", "ImageToClipTool", "CompositorTool", "SubtitleTool"]
