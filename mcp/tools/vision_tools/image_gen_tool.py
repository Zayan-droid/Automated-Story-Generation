"""Image-generation tool.

Provider precedence:
1. Pollinations.ai (free, no API key)
2. OpenAI image API if OPENAI_API_KEY is set and POLLINATIONS_DISABLE=1
3. Stable Diffusion (Automatic1111) if SD_API_URL is set
4. PIL placeholder (always works offline) — gradient + scene title overlay
"""
from __future__ import annotations
import hashlib
import os
import urllib.parse
from io import BytesIO
from pathlib import Path
from typing import Optional

from mcp.base_tool import BaseTool, ToolResult
from shared.utils.logging import get_logger

log = get_logger("image_gen")


class ImageGenTool(BaseTool):
    name = "vision.generate_image"
    description = "Generate an image from a text prompt (Pollinations / OpenAI / SD / PIL fallback)."
    category = "vision"

    def run(self, prompt: str, out_path: str, width: int = 1280, height: int = 720,
            seed: Optional[int] = None, style: str = "", negative_prompt: str = "",
            **_) -> ToolResult:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            out = out.with_suffix(".png")

        full_prompt = f"{prompt}, {style}".strip(", ") if style else prompt
        seed = seed or int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 2**31

        # 1. Pollinations.ai
        if os.getenv("POLLINATIONS_DISABLE") != "1":
            try:
                self._pollinations(full_prompt, out, width, height, seed)
                return ToolResult(success=True, data=str(out),
                                  metadata={"provider": "pollinations", "seed": seed})
            except Exception as e:  # noqa: BLE001
                log.warning("pollinations failed (%s) — trying next provider", e)

        # 2. OpenAI image gen
        if os.getenv("OPENAI_API_KEY"):
            try:
                self._openai_images(full_prompt, out, width, height)
                return ToolResult(success=True, data=str(out),
                                  metadata={"provider": "openai", "seed": seed})
            except Exception as e:  # noqa: BLE001
                log.warning("openai image gen failed (%s) — trying next provider", e)

        # 3. Stable Diffusion (Automatic1111-compatible)
        if os.getenv("SD_API_URL"):
            try:
                self._stable_diffusion(full_prompt, negative_prompt, out, width, height, seed)
                return ToolResult(success=True, data=str(out),
                                  metadata={"provider": "stable_diffusion", "seed": seed})
            except Exception as e:  # noqa: BLE001
                log.warning("stable diffusion failed (%s) — using PIL placeholder", e)

        # 4. PIL placeholder
        self._pil_placeholder(prompt, out, width, height, seed)
        return ToolResult(success=True, data=str(out),
                          metadata={"provider": "pil_placeholder", "seed": seed})

    # ---- providers -------------------------------------------------------

    def _pollinations(self, prompt: str, out: Path, w: int, h: int, seed: int) -> None:
        import requests
        encoded = urllib.parse.quote(prompt)
        url = (f"https://image.pollinations.ai/prompt/{encoded}"
               f"?width={w}&height={h}&seed={seed}&nologo=true&model=flux")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)

    def _openai_images(self, prompt: str, out: Path, w: int, h: int) -> None:
        from openai import OpenAI
        client = OpenAI()
        size = "1792x1024" if max(w, h) >= 1024 else "1024x1024"
        resp = client.images.generate(model="dall-e-3", prompt=prompt, size=size, n=1)
        import requests
        img_url = resp.data[0].url
        r = requests.get(img_url, timeout=60)
        out.write_bytes(r.content)

    def _stable_diffusion(self, prompt: str, negative: str, out: Path,
                          w: int, h: int, seed: int) -> None:
        import base64
        import requests
        api = os.environ["SD_API_URL"].rstrip("/")
        r = requests.post(
            f"{api}/sdapi/v1/txt2img",
            json={
                "prompt": prompt,
                "negative_prompt": negative,
                "width": w, "height": h,
                "seed": seed, "steps": 20, "sampler_index": "Euler a",
            },
            timeout=300,
        )
        r.raise_for_status()
        b64 = r.json()["images"][0]
        out.write_bytes(base64.b64decode(b64))

    def _pil_placeholder(self, prompt: str, out: Path, w: int, h: int, seed: int) -> None:
        from PIL import Image, ImageDraw, ImageFont
        import random
        rng = random.Random(seed)
        c1 = (rng.randint(20, 90), rng.randint(20, 90), rng.randint(40, 140))
        c2 = (rng.randint(120, 220), rng.randint(80, 200), rng.randint(60, 200))
        img = Image.new("RGB", (w, h), c1)
        draw = ImageDraw.Draw(img)
        # Vertical gradient.
        for y in range(h):
            t = y / max(1, h - 1)
            r = int(c1[0] + (c2[0] - c1[0]) * t)
            g = int(c1[1] + (c2[1] - c1[1]) * t)
            b = int(c1[2] + (c2[2] - c1[2]) * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))
        # Decorative shapes.
        for _ in range(8):
            x = rng.randint(0, w); yy = rng.randint(0, h)
            rad = rng.randint(40, 200)
            shade = (rng.randint(180, 255), rng.randint(180, 255), rng.randint(180, 255))
            draw.ellipse([x - rad, yy - rad, x + rad, yy + rad],
                         outline=shade, width=2)
        # Title.
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()
        text = (prompt[:80] + "…") if len(prompt) > 80 else prompt
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 20
        draw.rectangle(
            [(w - tw) // 2 - pad, h - th - 60 - pad,
             (w + tw) // 2 + pad, h - 60 + pad],
            fill=(0, 0, 0, 180),
        )
        draw.text(((w - tw) // 2, h - th - 60), text, fill=(255, 255, 255), font=font)
        img.save(out)
