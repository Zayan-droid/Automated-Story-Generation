"""TTS tool — gTTS primary, pyttsx3 fallback, silent placeholder as last resort."""
from __future__ import annotations
import os
import subprocess
from pathlib import Path

from mcp.base_tool import BaseTool, ToolResult
from shared.utils.logging import get_logger

log = get_logger("tts")


class TtsTool(BaseTool):
    name = "audio.tts"
    description = "Synthesize speech from text into a wav/mp3 file."
    category = "audio"

    def run(self, text: str, out_path: str, engine: str = "gtts",
            voice: str = "", language: str = "en", tld: str = "com",
            rate: int = 175, **_) -> ToolResult:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        engine = engine.lower()
        if engine == "edge":
            try:
                actual = self._edge_tts(text, out, voice)
                return ToolResult(success=True, data=str(actual), metadata={"engine": "edge"})
            except Exception as e:
                log.warning("edge-tts failed (%s) — falling back to gTTS", e)
                engine = "gtts"

        if engine == "elevenlabs" and os.getenv("ELEVENLABS_API_KEY"):
            try:
                actual = self._eleven_tts(text, out, voice)
                return ToolResult(success=True, data=str(actual), metadata={"engine": "elevenlabs"})
            except Exception as e:  # noqa: BLE001
                log.warning("elevenlabs failed (%s) — falling back to gTTS", e)
                engine = "gtts"

        if engine == "gtts":
            try:
                actual = self._gtts(text, out, language=language, tld=tld)
                return ToolResult(success=True, data=str(actual), metadata={"engine": "gtts"})
            except Exception as e:  # noqa: BLE001
                log.warning("gTTS failed (%s) — falling back to pyttsx3", e)
                engine = "pyttsx3"

        if engine == "pyttsx3":
            try:
                actual = self._pyttsx3(text, out, rate=rate, voice=voice)
                return ToolResult(success=True, data=str(actual), metadata={"engine": "pyttsx3"})
            except Exception as e:  # noqa: BLE001
                log.warning("pyttsx3 failed (%s) — using silent placeholder", e)

        # Last-ditch fallback: silent placeholder of approximate duration.
        duration = max(1.0, len(text.split()) / 2.5)
        actual = self._silent_wav(out, duration_s=duration)
        return ToolResult(success=True, data=str(actual),
                          metadata={"engine": "silent", "duration_s": duration})

    # ---- engines ---------------------------------------------------------

    def _gtts(self, text: str, out: Path, language: str = "en", tld: str = "com") -> Path:
        from gtts import gTTS
        if out.suffix.lower() != ".mp3":
            out = out.with_suffix(".mp3")
        gTTS(text=text, lang=language, tld=tld, slow=False).save(str(out))
        return out

    def _edge_tts(self, text: str, out: Path, voice: str = "") -> Path:
        voice_id = voice or "en-US-GuyNeural"
        if out.suffix.lower() != ".mp3":
            out = out.with_suffix(".mp3")
        cmd = ["edge-tts", "--voice", voice_id, "--text", text, "--write-media", str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        return out

    def _pyttsx3(self, text: str, out: Path, rate: int = 175, voice: str = "") -> Path:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        if voice:
            for v in engine.getProperty("voices"):
                if voice.lower() in (v.id or "").lower() or voice.lower() in (v.name or "").lower():
                    engine.setProperty("voice", v.id)
                    break
        if out.suffix.lower() != ".wav":
            out = out.with_suffix(".wav")
        engine.save_to_file(text, str(out))
        engine.runAndWait()
        engine.stop()
        return out

    def _eleven_tts(self, text: str, out: Path, voice: str = "") -> Path:
        import requests
        api_key = os.environ["ELEVENLABS_API_KEY"]
        voice_id = voice or os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        r = requests.post(
            url,
            headers={"xi-api-key": api_key, "Content-Type": "application/json"},
            json={"text": text, "model_id": "eleven_monolingual_v1"},
            timeout=60,
        )
        r.raise_for_status()
        if out.suffix.lower() != ".mp3":
            out = out.with_suffix(".mp3")
        out.write_bytes(r.content)
        return out

    def _silent_wav(self, out: Path, duration_s: float = 2.0) -> Path:
        if out.suffix.lower() != ".wav":
            out = out.with_suffix(".wav")
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=channel_layout=mono:sample_rate=22050",
            "-t", f"{duration_s:.2f}",
            "-q:a", "9", "-acodec", "pcm_s16le",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return out
