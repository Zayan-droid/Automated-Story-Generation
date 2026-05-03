"""Cinematic animator — turns stills into watchable video.

Two key upgrades over the basic ken-burns clip:

1. **Multi-shot scene composition.** Each scene gets one establishing shot
   image plus one portrait per character that speaks in it. Each dialogue
   line becomes its OWN sub-clip (shot of the speaker if it's a character,
   or wide shot for the narrator), and sub-clips are crossfaded together
   inside the scene. Result: a 4-scene project becomes ~10-15 cuts instead
   of 4 long static shots.

2. **Smoother motion.** We use a richer ffmpeg filter chain — high-resolution
   source crop, slow zoompan with eased curves, optional `minterpolate` for
   frame interpolation, subtle vignette + film-grain for cinematic feel.
"""
from __future__ import annotations
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# Distinct motion patterns that look good on a still image.
MOTION_PRESETS = [
    "slow_zoom_in",
    "slow_zoom_out",
    "pan_left_subtle",
    "pan_right_subtle",
    "ken_burns_diag",
    "ken_burns_diag_rev",
    "static_breathing",
]


@dataclass
class Shot:
    """A single sub-clip inside a scene."""
    image_path: str
    duration_ms: int
    motion: str = "ken_burns_diag"
    audio_path: Optional[str] = None       # if set, gets muxed in (used for lip sync)
    is_lip_sync: bool = False              # if True, image_path is already a video


def probe_duration_ms(path: str | Path) -> Optional[int]:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        return int(float(json.loads(r.stdout)["format"]["duration"]) * 1000)
    except Exception:  # noqa: BLE001
        return None


def render_shot(shot: Shot, out_path: Path, width: int, height: int, fps: int,
                add_grain: bool = True, add_vignette: bool = True) -> Path:
    """Render a single shot (still image -> mp4) with cinematic motion."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")

    if shot.is_lip_sync and shot.image_path.lower().endswith((".mp4", ".mov", ".webm")):
        # Already a video — just normalize size/format.
        return _normalize_video(shot.image_path, out_path, width, height, fps,
                                duration_ms=shot.duration_ms,
                                audio_path=shot.audio_path,
                                add_grain=add_grain, add_vignette=add_vignette)

    dur_s = max(0.5, shot.duration_ms / 1000.0)
    frames = max(1, int(round(dur_s * fps)))
    motion_filter = _motion_filter_for(shot.motion, frames, width, height)

    # Cinematic post-chain: subtle vignette + grain + colour grading nudge.
    post = []
    if add_vignette:
        post.append("vignette=PI/5")
    if add_grain:
        post.append("noise=alls=4:allf=t+u")
    # Colour grading: very mild S-curve.
    post.append("eq=contrast=1.04:saturation=1.06")
    post.append("format=yuv420p")
    post_chain = "," + ",".join(post)

    vf = (
        f"scale=w={int(width*1.6)}:h={int(height*1.6)}:force_original_aspect_ratio=increase,"
        f"crop={int(width*1.6)}:{int(height*1.6)},"
        f"{motion_filter}"
        f"{post_chain}"
    )

    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", str(shot.image_path)]
    if shot.audio_path and Path(shot.audio_path).exists():
        cmd += ["-i", str(shot.audio_path)]
    cmd += [
        "-vf", vf,
        "-t", f"{dur_s:.3f}",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p",
    ]
    if shot.audio_path and Path(shot.audio_path).exists():
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def assemble_scene(shots: List[Path], out_path: Path,
                   crossfade_ms: int = 250,
                   audio_path: Optional[str] = None) -> Path:
    """Concatenate the sub-clips of a scene with crossfades and (optional) audio."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")

    if len(shots) == 1:
        # Single shot — just normalize.
        return _normalize_video(str(shots[0]), out_path,
                                duration_ms=probe_duration_ms(shots[0]) or 3000,
                                audio_path=audio_path, add_grain=False,
                                add_vignette=False, width=0, height=0, fps=24)

    # Probe per-clip durations so we can compute xfade offsets.
    durs = [probe_duration_ms(s) or 1000 for s in shots]
    xfade_s = max(0.05, crossfade_ms / 1000.0)

    inputs: List[str] = []
    for s in shots:
        inputs += ["-i", str(s)]
    if audio_path and Path(audio_path).exists():
        inputs += ["-i", str(audio_path)]

    filter_parts = []
    prev = "0:v"
    offset = 0.0
    for i in range(1, len(shots)):
        offset += (durs[i - 1] / 1000.0) - xfade_s
        out_label = f"v{i}"
        filter_parts.append(
            f"[{prev}][{i}:v]xfade=transition=fade:duration={xfade_s:.2f}:"
            f"offset={max(0,offset):.3f}[{out_label}]"
        )
        prev = out_label
    fc = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", fc,
           "-map", f"[{prev}]"]
    if audio_path and Path(audio_path).exists():
        cmd += ["-map", f"{len(shots)}:a:0", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


# ---- internals --------------------------------------------------------------

def _motion_filter_for(motion: str, frames: int, w: int, h: int) -> str:
    """Return a `zoompan=...` filter string for the requested motion."""
    f = max(1, frames)
    sw, sh = w, h
    if motion == "slow_zoom_in":
        return (f"zoompan=z='min(zoom+0.0010,1.30)':d={f}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={sw}x{sh}:fps={f/(f/24)}")
    if motion == "slow_zoom_out":
        return (f"zoompan=z='if(eq(on,0),1.30,max(zoom-0.0010,1.0))':d={f}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={sw}x{sh}")
    if motion == "pan_left_subtle":
        return (f"zoompan=z=1.18:x='if(eq(on,0),iw-iw/zoom,max(0,x-1.5))':"
                f"y='ih/2-(ih/zoom/2)':d={f}:s={sw}x{sh}")
    if motion == "pan_right_subtle":
        return (f"zoompan=z=1.18:x='if(eq(on,0),0,min(iw-iw/zoom,x+1.5))':"
                f"y='ih/2-(ih/zoom/2)':d={f}:s={sw}x{sh}")
    if motion == "ken_burns_diag":
        return (f"zoompan=z='min(zoom+0.0008,1.22)':d={f}:"
                f"x='if(eq(on,0),0,min(iw-iw/zoom,x+0.6))':"
                f"y='if(eq(on,0),0,min(ih-ih/zoom,y+0.4))':s={sw}x{sh}")
    if motion == "ken_burns_diag_rev":
        return (f"zoompan=z='min(zoom+0.0008,1.22)':d={f}:"
                f"x='if(eq(on,0),iw-iw/zoom,max(0,x-0.6))':"
                f"y='if(eq(on,0),ih-ih/zoom,max(0,y-0.4))':s={sw}x{sh}")
    if motion == "static_breathing":
        return (f"zoompan=z='1.0+0.015*sin(2*PI*on/{max(1,f)})':d={f}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={sw}x{sh}")
    # default
    return (f"zoompan=z='min(zoom+0.0008,1.22)':d={f}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={sw}x{sh}")


def _normalize_video(in_path: str, out_path: Path, width: int, height: int,
                     fps: int, duration_ms: int,
                     audio_path: Optional[str] = None,
                     add_grain: bool = False, add_vignette: bool = False) -> Path:
    """Re-encode an existing video clip to the project's pixel format / size."""
    dur_s = max(0.5, duration_ms / 1000.0)
    vf_parts = []
    if width and height:
        vf_parts.append(
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}"
        )
    if add_vignette:
        vf_parts.append("vignette=PI/5")
    if add_grain:
        vf_parts.append("noise=alls=4:allf=t+u")
    vf_parts.append("format=yuv420p")
    if fps:
        vf_parts.append(f"fps={fps}")
    vf = ",".join(vf_parts)

    cmd = ["ffmpeg", "-y", "-i", str(in_path)]
    if audio_path and Path(audio_path).exists():
        cmd += ["-i", str(audio_path)]
    cmd += ["-vf", vf, "-t", f"{dur_s:.3f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p"]
    if audio_path and Path(audio_path).exists():
        cmd += ["-map", "0:v:0", "-map", "1:a:0",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-c:a", "copy" if _has_audio(in_path) else "aac",
                "-shortest"]
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def _has_audio(path: str | Path) -> bool:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_type", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        return bool(json.loads(r.stdout).get("streams"))
    except Exception:  # noqa: BLE001
        return False


def pick_motion_for_index(scene_idx: int, shot_idx: int) -> str:
    """Deterministic motion picker — alternates so adjacent shots feel different."""
    n = len(MOTION_PRESETS)
    return MOTION_PRESETS[(scene_idx * 3 + shot_idx) % n]
