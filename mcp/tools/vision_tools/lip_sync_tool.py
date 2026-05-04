"""Lip-sync tool — turns a portrait + audio into a talking-head clip.

Provider precedence:

1. fal.ai SadTalker (real ML lip sync — set FAL_KEY)
   Uses the official fal-client SDK.
   Models tried in order:
     - "fal-ai/sadtalker"
     - "fal-ai/sync-lipsync"

2. Replicate Wav2Lip (set REPLICATE_API_TOKEN — free $0.01 trial credit)
   Uses the official replicate Python SDK.
   Models tried in order:
     - "devxpy/cog-wav2lip"   — best Wav2Lip quality
     - "lucataco/wav2lip"     — alternative
     - "lucataco/sadtalker"   — fallback if Wav2Lip quota exhausted

3. Heuristic fallback — zoompan mouth-region pulse, fully offline.
   Visibly "active" during dialogue — no ML required.
"""
from __future__ import annotations
import base64
import os
import subprocess
from pathlib import Path
from typing import Optional

from mcp.base_tool import BaseTool, ToolResult
from shared.utils.logging import get_logger

log = get_logger("lip_sync")

_FAL_LS_MODELS = [
    "fal-ai/sadtalker",
    "fal-ai/sync-lipsync",
]

# Replicate Wav2Lip — only `devxpy/cog-wav2lip` is currently public.
# Note: free Replicate accounts are rate-limited to 6 req/min with burst=1
# until a payment method is added. Add one at replicate.com/account/billing.
_REPLICATE_LS_MODELS = [
    "devxpy/cog-wav2lip",
]


class LipSyncTool(BaseTool):
    name = "vision.lip_sync"
    description = "Generate a talking-head clip from a portrait image + audio file."
    category = "vision"

    def run(self, image_path: str, audio_path: str, out_path: str,
            duration_s: Optional[float] = None,
            width: int = 1280, height: int = 720, fps: int = 24,
            **_) -> ToolResult:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() != ".mp4":
            out = out.with_suffix(".mp4")

        if not Path(image_path).exists():
            return ToolResult(success=False, error=f"missing image: {image_path}")
        if not Path(audio_path).exists():
            return ToolResult(success=False, error=f"missing audio: {audio_path}")

        # ---- provider 1: fal.ai -----------------------------------------
        if os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY"):
            try:
                provider = self._fal_sadtalker(image_path, audio_path, out)
                return ToolResult(success=True, data=str(out),
                                  metadata={"provider": provider})
            except Exception as e:  # noqa: BLE001
                log.warning("fal lip-sync failed (%s) — trying Replicate", e)

        # ---- provider 2: Replicate Wav2Lip ------------------------------
        if os.getenv("REPLICATE_API_TOKEN"):
            try:
                provider = self._replicate_wav2lip(image_path, audio_path, out,
                                                   width=width, height=height, fps=fps)
                return ToolResult(success=True, data=str(out),
                                  metadata={"provider": provider})
            except Exception as e:  # noqa: BLE001
                log.warning("Replicate lip-sync failed (%s) — using heuristic", e)

        # ---- provider 3: heuristic offline fallback ---------------------
        self._heuristic_talking_head(image_path, audio_path, out,
                                     width=width, height=height, fps=fps)
        return ToolResult(success=True, data=str(out),
                          metadata={"provider": "heuristic"})

    # ---- fal.ai ---------------------------------------------------------

    def _fal_sadtalker(self, image_path: str, audio_path: str, out: Path) -> str:
        import fal_client
        import requests

        def _to_data_url(path: str, default_mime: str) -> str:
            suffix = Path(path).suffix.lower()
            mime_map = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".wav": "audio/wav", ".mp3": "audio/mpeg",
            }
            mime = mime_map.get(suffix, default_mime)
            with open(path, "rb") as f:
                return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

        img_url = _to_data_url(image_path, "image/png")
        aud_url = _to_data_url(audio_path, "audio/wav")

        last_err: Exception | None = None
        for model_id in _FAL_LS_MODELS:
            try:
                log.info("fal: submitting %s ...", model_id)
                args = (
                    {"source_image_url": img_url, "driven_audio_url": aud_url,
                     "preprocess": "full", "still_mode": False, "use_enhancer": False}
                    if "sadtalker" in model_id
                    else {"video_url": img_url, "audio_url": aud_url,
                          "model": "lipsync-1.7.1", "sync_mode": "bounce"}
                )
                result = fal_client.subscribe(model_id, arguments=args, with_logs=False)
                vid = result.get("video", {})
                url = (vid.get("url") if isinstance(vid, dict)
                       else result.get("video_url") or result.get("url"))
                if url:
                    r = requests.get(url, timeout=300)
                    r.raise_for_status()
                    out.write_bytes(r.content)
                    log.info("fal: lip sync done via %s", model_id)
                    return f"fal_{model_id.split('/')[-1]}"
                raise RuntimeError(f"no video URL in response: {str(result)[:200]}")
            except Exception as e:  # noqa: BLE001
                log.warning("fal %s failed: %s", model_id, e)
                last_err = e
        raise RuntimeError(f"all fal models failed; last: {last_err}")

    # ---- Replicate Wav2Lip ----------------------------------------------

    def _replicate_wav2lip(self, image_path: str, audio_path: str, out: Path,
                           width: int, height: int, fps: int) -> str:
        """Run Wav2Lip (or SadTalker fallback) via the Replicate Python SDK.

        The SDK reads REPLICATE_API_TOKEN from env automatically, handles
        file uploads, queue submission, and polling.
        """
        import replicate
        import requests

        last_err: Exception | None = None
        for model_slug in _REPLICATE_LS_MODELS:
            try:
                log.info("replicate: submitting %s ...", model_slug)

                with open(image_path, "rb") as img_f, \
                        open(audio_path, "rb") as aud_f:

                    if "wav2lip" in model_slug:
                        # cog-wav2lip / lucataco/wav2lip schema
                        inp = {
                            "face": img_f,
                            "audio": aud_f,
                            "pads": "0 10 0 0",   # padding: top right bottom left
                            "fps": min(fps, 25),
                            "out_height": min(height, 720),
                            "smooth": True,
                            "resize_factor": 1,
                        }
                    else:
                        # lucataco/sadtalker schema
                        inp = {
                            "source_image": img_f,
                            "driven_audio": aud_f,
                            "preprocess": "full",
                            "still_mode": False,
                            "use_enhancer": False,
                            "pose_style": 0,
                            "size": 256,
                            "expression_scale": 1.0,
                        }

                    output = replicate.run(model_slug, input=inp)

                # Output is a URL string or list of URLs.
                url = output if isinstance(output, str) else (
                    output[0] if isinstance(output, list) else None
                )
                if not url:
                    raise RuntimeError(f"no output URL from {model_slug}: {output}")

                log.info("replicate: downloading result from %s ...", model_slug)
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                out.write_bytes(r.content)
                log.info("replicate: lip sync complete via %s", model_slug)
                return f"replicate_{model_slug.split('/')[-1]}"

            except Exception as e:  # noqa: BLE001
                log.warning("replicate %s failed: %s", model_slug, e)
                last_err = e

        raise RuntimeError(f"all Replicate lip-sync models failed; last: {last_err}")

    # ---- heuristic offline fallback ------------------------------------

    def _heuristic_talking_head(self, image_path: str, audio_path: str, out: Path,
                                width: int, height: int, fps: int) -> None:
        """Zoompan mouth-region pulse simulating speech — works 100% offline."""
        import json as _json
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(audio_path)],
            capture_output=True, text=True, check=True,
        )
        duration_s = max(0.5, float(_json.loads(probe.stdout)["format"]["duration"]))

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(image_path),
            "-i", str(audio_path),
            "-filter_complex",
            (
                f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},"
                f"zoompan=z='1.0+0.04*abs(sin(2*PI*on/12))':"
                f"d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)*1.05':"
                f"s={width}x{height}:fps={fps}[v]"
            ),
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-t", f"{duration_s:.2f}",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
