"""Background music generator — produces simple mood-appropriate tones via ffmpeg."""
from __future__ import annotations
import subprocess
from pathlib import Path

from mcp.base_tool import BaseTool, ToolResult


# Mood -> (base_freq_hz, second_freq_hz, harmonic_freq_hz, volume)
MOOD_PRESETS = {
    "ambient":   (220, 277, 440, 0.18),
    "tense":     (110, 138, 220, 0.20),
    "joyful":    (440, 554, 880, 0.22),
    "mysterious":(165, 207, 330, 0.18),
    "epic":      (110, 165, 220, 0.25),
    "sad":       (146, 184, 293, 0.18),
    "ominous":   ( 87, 110, 174, 0.22),
    "ethereal":  (392, 493, 783, 0.20),
    "energetic": (293, 369, 587, 0.24),
    "neutral":   (220, 277, 440, 0.18),
}


class BgmTool(BaseTool):
    name = "audio.bgm"
    description = "Generate ambient background music for a given mood + duration."
    category = "audio"

    def run(self, mood: str, duration_ms: int, out_path: str, **_) -> ToolResult:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mood_key = mood.lower().strip() or "ambient"
        preset = MOOD_PRESETS.get(mood_key, MOOD_PRESETS["ambient"])
        f1, f2, f3, vol = preset
        dur_s = max(0.5, duration_ms / 1000.0)

        # Layered sines + low-pass + tremolo, exported as wav.
        if out.suffix.lower() not in (".wav", ".mp3"):
            out = out.with_suffix(".wav")

        filter_complex = (
            f"sine=frequency={f1}:duration={dur_s},"
            f"volume={vol},"
            f"tremolo=f=4:d=0.3,"
            f"aformat=sample_fmts=s16:sample_rates=22050:channel_layouts=mono [a1];"
            f"sine=frequency={f2}:duration={dur_s},"
            f"volume={vol*0.7},"
            f"aformat=sample_fmts=s16:sample_rates=22050:channel_layouts=mono [a2];"
            f"sine=frequency={f3}:duration={dur_s},"
            f"volume={vol*0.4},"
            f"aformat=sample_fmts=s16:sample_rates=22050:channel_layouts=mono [a3];"
            f"[a1][a2][a3]amix=inputs=3:normalize=0,"
            f"afade=t=in:st=0:d=0.5,"
            f"afade=t=out:st={max(0, dur_s-0.5):.2f}:d=0.5"
        )

        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "anullsrc=channel_layout=mono:sample_rate=22050",
            "-filter_complex", filter_complex,
            "-t", f"{dur_s:.2f}",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return ToolResult(success=True, data=str(out),
                          metadata={"mood": mood_key, "duration_ms": duration_ms})
