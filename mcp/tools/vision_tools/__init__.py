"""Vision tools — image generation, editing, style transfer, video synthesis, lip sync."""
from .image_gen_tool import ImageGenTool
from .image_edit_tool import ImageEditTool
from .style_transfer import StyleTransferTool
from .text_to_video_tool import TextToVideoTool
from .lip_sync_tool import LipSyncTool
from mcp.tool_registry import registry

registry.register(ImageGenTool())
registry.register(ImageEditTool())
registry.register(StyleTransferTool())
registry.register(TextToVideoTool())
registry.register(LipSyncTool())

__all__ = ["ImageGenTool", "ImageEditTool", "StyleTransferTool",
           "TextToVideoTool", "LipSyncTool"]
