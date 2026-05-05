"""Subtitle overlay — burn-in SRT subtitles via ffmpeg."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Dict, Any

from mcp.base_tool import BaseTool, ToolResult


def _ms_to_srt_ts(ms: int) -> str:
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, msec = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"


class SubtitleTool(BaseTool):
    name = "video.subtitle"
    description = "Burn dialogue subtitles from a list of {start_ms,end_ms,text} dicts onto an MP4."
    category = "video"

    def run(self, in_path: str, out_path: str, lines: List[Dict[str, Any]],
            font_size: int = 20, **_) -> ToolResult:
        in_p = Path(in_path)
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        srt_p = out_p.with_suffix(".srt")
        srt_p.write_text(self._build_srt(lines), encoding="utf-8")
        # ffmpeg's subtitles filter wants forward slashes even on Windows.
        srt_arg = srt_p.resolve().as_posix().replace(":", "\\:")
        # BorderStyle=1 = outline only (no opaque box).
        # Alignment=2 = bottom-center (ASS standard).
        # MarginV=40 keeps lines off the very edge.
        # WrapStyle=2 = no automatic line breaks unless we add \N (forces single line then wraps to 2).
        vf = (f"subtitles=filename='{srt_arg}':"
              f"force_style='FontSize={font_size},"
              f"PrimaryColour=&Hffffff&,OutlineColour=&H000000&,BackColour=&H80000000&,"
              f"BorderStyle=1,Outline=2,Shadow=1,"
              f"Alignment=2,MarginV=40,MarginL=80,MarginR=80,"
              f"Bold=0'")
        cmd = ["ffmpeg", "-y", "-i", str(in_p), "-vf", vf,
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
               "-c:a", "copy", str(out_p)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return ToolResult(success=False, error=proc.stderr[-2000:],
                              metadata={"cmd": " ".join(cmd)})
        return ToolResult(success=True, data=str(out_p),
                          metadata={"srt": str(srt_p), "line_count": len(lines)})

    @staticmethod
    def _build_srt(lines: List[Dict[str, Any]]) -> str:
        chunks = []
        for i, ln in enumerate(lines, start=1):
            chunks.append(
                f"{i}\n"
                f"{_ms_to_srt_ts(int(ln['start_ms']))} --> "
                f"{_ms_to_srt_ts(int(ln['end_ms']))}\n"
                f"{ln['text'].strip()}\n"
            )
        return "\n".join(chunks)
