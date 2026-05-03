"""Audio merging — concatenate segments and overlay BGM via ffmpeg."""
from __future__ import annotations
import subprocess
from pathlib import Path
from typing import List, Optional

from mcp.base_tool import BaseTool, ToolResult
from shared.utils.logging import get_logger

log = get_logger("audio_merger")


class AudioMergerTool(BaseTool):
    name = "audio.merge"
    description = (
        "Concatenate dialogue clips into a master track and optionally mix in a BGM bed."
    )
    category = "audio"

    def run(self, segments: List[str], out_path: str, bgm: Optional[str] = None,
            bgm_volume: float = 0.25, **_) -> ToolResult:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() not in (".wav", ".mp3", ".m4a"):
            out = out.with_suffix(".wav")

        segs = [s for s in segments if Path(s).exists()]
        if not segs:
            return ToolResult(success=False, error="no input segments exist")

        # Concatenate dialogue segments.
        concat_path = out.parent / f"{out.stem}_dialogue.wav"
        self._concat(segs, concat_path)

        if bgm and Path(bgm).exists():
            self._mix(concat_path, Path(bgm), out, bgm_volume=bgm_volume)
        else:
            # Re-encode straight to the target format.
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(concat_path), str(out)],
                check=True, capture_output=True,
            )

        try:
            concat_path.unlink()
        except Exception:  # noqa: BLE001
            pass

        return ToolResult(success=True, data=str(out),
                          metadata={"segment_count": len(segs), "bgm": bool(bgm)})

    # ---- helpers ---------------------------------------------------------

    def _concat(self, segs: List[str], out: Path) -> None:
        list_path = out.parent / f"{out.stem}_concat.txt"
        list_path.write_text(
            "\n".join(f"file '{Path(s).resolve().as_posix()}'" for s in segs),
            encoding="utf-8",
        )
        # Decode to PCM WAV first so the concat demuxer never trips on mixed codecs/headers.
        norm_dir = out.parent / "_norm"
        norm_dir.mkdir(exist_ok=True)
        norm_paths: List[Path] = []
        for i, s in enumerate(segs):
            norm = norm_dir / f"seg_{i:03d}.wav"
            subprocess.run(
                ["ffmpeg", "-y", "-i", s,
                 "-ar", "22050", "-ac", "1", "-c:a", "pcm_s16le",
                 str(norm)],
                check=True, capture_output=True,
            )
            norm_paths.append(norm)

        list_path.write_text(
            "\n".join(f"file '{p.resolve().as_posix()}'" for p in norm_paths),
            encoding="utf-8",
        )
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(list_path), "-c", "copy", str(out)],
            check=True, capture_output=True,
        )

        # cleanup
        try:
            for p in norm_paths:
                p.unlink(missing_ok=True)
            norm_dir.rmdir()
            list_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    def _mix(self, dialogue: Path, bgm: Path, out: Path, bgm_volume: float = 0.25) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(dialogue),
            "-i", str(bgm),
            "-filter_complex",
            f"[1:a]volume={bgm_volume},aloop=loop=-1:size=2e9[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[a]",
            "-map", "[a]",
            "-ar", "22050", "-ac", "2",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
