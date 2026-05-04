"""Text-to-video / image-to-video tool.

Real video generation. Provider precedence:

1. fal.ai (recommended — free trial credits, simplest API)
   Uses the official fal-client SDK which handles auth + queue automatically.
   Models tried in order:
     - "fal-ai/stable-video-diffusion"  image -> video (4 s clip)
     - "fal-ai/fast-animatediff/text-to-video"  text -> video fallback

2. Replicate (paid; requires REPLICATE_API_TOKEN)
   - "stability-ai/stable-video-diffusion"

3. Hugging Face Inference API (free tier; lower quality)
   - "damo-vilab/text-to-video-ms-1.7b"

If none are configured, this tool returns ToolResult(success=False) so the
caller falls back to the ffmpeg ken-burns animator.
"""
from __future__ import annotations
import base64
import os
from pathlib import Path
from typing import Optional

from mcp.base_tool import BaseTool, ToolResult
from shared.utils.logging import get_logger

log = get_logger("text_to_video")

# fal.ai image-to-video model IDs to try in order
_FAL_I2V_MODELS = [
    "fal-ai/stable-video-diffusion",
    "fal-ai/fast-animatediff/turbo",
]
# fal.ai text-to-video model IDs to try in order
_FAL_T2V_MODELS = [
    "fal-ai/fast-animatediff/text-to-video",
    "fal-ai/cogvideox-5b",
]


class TextToVideoTool(BaseTool):
    name = "vision.text_to_video"
    description = "Generate a real video clip from text + (optional) source image."
    category = "vision"

    def run(self, prompt: str, out_path: str,
            image_path: Optional[str] = None,
            duration_s: float = 4.0,
            width: int = 1024, height: int = 576,
            fps: int = 24, **_) -> ToolResult:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() != ".mp4":
            out = out.with_suffix(".mp4")

        if os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY"):
            try:
                self._fal(prompt, image_path, out, duration_s, width, height, fps)
                return ToolResult(success=True, data=str(out),
                                  metadata={"provider": "fal", "duration_s": duration_s})
            except Exception as e:  # noqa: BLE001
                log.warning("fal.ai text->video failed (%s) — trying next provider", e)

        if os.getenv("REPLICATE_API_TOKEN"):
            try:
                self._replicate(prompt, image_path, out, duration_s, width, height)
                return ToolResult(success=True, data=str(out),
                                  metadata={"provider": "replicate"})
            except Exception as e:  # noqa: BLE001
                log.warning("replicate text->video failed (%s) — trying next provider", e)

        if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY"):
            try:
                self._huggingface(prompt, out, duration_s, width, height)
                return ToolResult(success=True, data=str(out),
                                  metadata={"provider": "huggingface"})
            except Exception as e:  # noqa: BLE001
                log.warning("hf text->video failed (%s) — no provider available", e)

        return ToolResult(success=False,
                          error="no text-to-video provider configured "
                                "(set FAL_KEY, REPLICATE_API_TOKEN, or HF_TOKEN)")

    # ---- providers -------------------------------------------------------

    def _fal(self, prompt: str, image_path: Optional[str], out: Path,
             duration_s: float, w: int, h: int, fps: int) -> None:
        """Use fal-client SDK with base64 data URLs (no CDN upload required)."""
        import fal_client
        import requests

        def _download(url: str) -> None:
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            out.write_bytes(r.content)

        def _to_data_url(path: str) -> str:
            mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
            with open(path, "rb") as f:
                return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"

        def _extract_url(result: dict) -> Optional[str]:
            vid = result.get("video", {})
            return (vid.get("url") if isinstance(vid, dict)
                    else result.get("video_url") or result.get("url"))

        if image_path and Path(image_path).exists():
            img_data_url = _to_data_url(image_path)
            last_err: Exception | None = None
            for model_id in _FAL_I2V_MODELS:
                try:
                    log.info("fal: submitting i2v %s ...", model_id)
                    result = fal_client.subscribe(
                        model_id,
                        arguments={
                            "image_url": img_data_url,
                            "motion_bucket_id": 100,
                            "cond_aug": 0.02,
                            "fps": min(fps, 24),
                            "num_frames": int(duration_s * min(fps, 24)),
                        },
                        with_logs=False,
                    )
                    url = _extract_url(result)
                    if url:
                        _download(url)
                        log.info("fal: i2v done via %s", model_id)
                        return
                    raise RuntimeError(f"no video URL in response: {str(result)[:200]}")
                except Exception as e:  # noqa: BLE001
                    log.warning("fal %s i2v failed: %s", model_id, e)
                    last_err = e
            raise RuntimeError(f"all fal i2v models failed; last: {last_err}")
        else:
            last_err = None
            for model_id in _FAL_T2V_MODELS:
                try:
                    log.info("fal: submitting t2v %s ...", model_id)
                    result = fal_client.subscribe(
                        model_id,
                        arguments={
                            "prompt": prompt,
                            "video_size": {"width": w, "height": h},
                            "num_frames": int(duration_s * min(fps, 24)),
                            "fps": min(fps, 24),
                        },
                        with_logs=False,
                    )
                    url = _extract_url(result)
                    if url:
                        _download(url)
                        log.info("fal: t2v done via %s", model_id)
                        return
                    raise RuntimeError(f"no video URL in response: {str(result)[:200]}")
                except Exception as e:  # noqa: BLE001
                    log.warning("fal %s t2v failed: %s", model_id, e)
                    last_err = e
            raise RuntimeError(f"all fal t2v models failed; last: {last_err}")

    def _replicate(self, prompt: str, image_path: Optional[str], out: Path,
                   duration_s: float, w: int, h: int) -> None:
        """Run text-to-video / image-to-video via the Replicate Python SDK."""
        import replicate
        import requests

        if image_path and Path(image_path).exists():
            log.info("replicate: submitting stability-ai/stable-video-diffusion ...")
            with open(image_path, "rb") as img_f:
                output = replicate.run(
                    "stability-ai/stable-video-diffusion",
                    input={
                        "input_image": img_f,
                        "video_length": "25_frames_with_svd_xt",
                        "fps_id": 6,
                        "motion_bucket_id": 127,
                        "cond_aug": 0.02,
                        "decoding_t": 7,
                    },
                )
        else:
            log.info("replicate: submitting lucataco/zeroscope-v2-xl (t2v) ...")
            output = replicate.run(
                "lucataco/zeroscope-v2-xl:9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351",
                input={
                    "prompt": prompt,
                    "num_frames": max(16, int(duration_s * 8)),
                    "fps": 8,
                    "width": min(w, 1024),
                    "height": min(h, 576),
                    "num_inference_steps": 40,
                },
            )

        url = output if isinstance(output, str) else (
            output[0] if isinstance(output, list) else None
        )
        if not url:
            raise RuntimeError(f"no output URL from replicate: {output}")

        log.info("replicate: downloading video ...")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        out.write_bytes(r.content)

    def _huggingface(self, prompt: str, out: Path, duration_s: float,
                     w: int, h: int) -> None:
        import requests
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        headers = {"Authorization": f"Bearer {token}"}
        url = "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"
        r = requests.post(url, headers=headers,
                          json={"inputs": prompt}, timeout=180)
        r.raise_for_status()
        out.write_bytes(r.content)
