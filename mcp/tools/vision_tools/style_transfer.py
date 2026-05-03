"""Style transfer — light-weight stylization via filter combinations.

Real neural style transfer is heavy; for a student project we map named
'styles' to deterministic filter chains so it always works offline.
"""
from __future__ import annotations
from pathlib import Path

from mcp.base_tool import BaseTool, ToolResult
from .image_edit_tool import FILTERS


STYLES = {
    "cinematic": [("contrast", {"factor": 1.25}),
                  ("saturation", {"factor": 0.85}),
                  ("warm", {})],
    "noir": [("grayscale", {}), ("contrast", {"factor": 1.4})],
    "dreamy": [("blur", {"radius": 2.0}),
               ("brightness", {"factor": 1.15}),
               ("saturation", {"factor": 1.2})],
    "anime": [("saturation", {"factor": 1.5}),
              ("sharpness", {"factor": 2.0}),
              ("contrast", {"factor": 1.2})],
    "pastel": [("brightness", {"factor": 1.15}),
               ("saturation", {"factor": 0.7}),
               ("warm", {})],
    "vintage": [("vintage", {})],
    "cold_thriller": [("cool", {}),
                      ("contrast", {"factor": 1.3}),
                      ("saturation", {"factor": 0.6})],
}


class StyleTransferTool(BaseTool):
    name = "vision.style_transfer"
    description = "Apply a named style preset (cinematic/noir/anime/...) to an image."
    category = "vision"

    def run(self, in_path: str, out_path: str, style: str, **_) -> ToolResult:
        from PIL import Image
        chain = STYLES.get(style.lower())
        if not chain:
            return ToolResult(success=False,
                              error=f"unknown style '{style}'; available: {list(STYLES)}")
        img = Image.open(in_path).convert("RGB")
        for name, params in chain:
            fn = FILTERS[name]
            img = fn(img, **params) if params else fn(img)
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_p)
        return ToolResult(success=True, data=str(out_p),
                          metadata={"style": style, "filters": [f for f, _ in chain]})


def list_style_names() -> list[str]:
    return sorted(STYLES.keys())
