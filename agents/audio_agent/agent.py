"""Phase 2 agent: ScriptOutput -> AudioOutput.

Per-line TTS with character-consistent voices, per-scene BGM, then a single
master track. Emits both the structured AudioOutput and a flat
timing_manifest.json the spec calls for.
"""
from __future__ import annotations
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mcp.tool_executor import ToolExecutor
from shared.constants import PHASE_AUDIO
from shared.schemas.audio import (
    AudioOutput, AudioSegment, TimingManifest, VoiceConfig,
)
from shared.schemas.pipeline import PipelineState
from shared.schemas.story import Character, ScriptOutput
from shared.utils.files import asset_path, project_dir, write_json
from shared.utils.logging import get_logger

log = get_logger("audio_agent")


class AudioAgent:
    def __init__(self):
        self.tools = ToolExecutor()

    # ---- public ----------------------------------------------------------

    def run(self, state: PipelineState, with_bgm: bool = True,
            tts_engine: str = "gtts") -> AudioOutput:
        if not state.script:
            raise ValueError("phase 2 requires state.script (run phase 1 first)")
        log.info("phase 2 start (project=%s, engine=%s)", state.project_id, tts_engine)
        state.phase2.status = "running"
        state.phase2.started_at = datetime.utcnow().isoformat()

        try:
            voice_configs = self._build_voice_configs(state.script, default_engine=tts_engine)
            segments, manifest = self._render_segments(state, voice_configs)
            bgm_path: Optional[str] = None
            master_path: Optional[str] = None
            if with_bgm:
                bgm_path = self._render_bgm(state)
            master_path = self._mix_master(state, segments, bgm_path)

            output = AudioOutput(
                voice_configs=voice_configs,
                manifest=manifest,
                bgm_track=bgm_path,
                master_track=master_path,
            )
            artifacts = self._serialize(state.project_id, output)
            state.audio = output
            state.phase2.status = "complete"
            state.phase2.finished_at = datetime.utcnow().isoformat()
            state.phase2.artifact_paths = artifacts
            log.info("phase 2 complete (%d dialogue segments, master=%s)",
                     len(segments), master_path)
            return output
        except Exception as e:  # noqa: BLE001
            state.phase2.status = "failed"
            state.phase2.error = f"{type(e).__name__}: {e}"
            log.exception("phase 2 failed")
            raise

    # ---- voice configs ---------------------------------------------------

    def _build_voice_configs(self, script: ScriptOutput, default_engine: str) -> List[VoiceConfig]:
        out: List[VoiceConfig] = []
        for c in script.characters.characters:
            tld = self._tld_for(c)
            rate = self._rate_for(c)
            out.append(VoiceConfig(
                character_id=c.id,
                engine=default_engine,
                language="en",
                tld=tld,
                rate=rate,
                tone=c.voice_style,
            ))
        return out

    @staticmethod
    def _tld_for(c: Character) -> str:
        # Different TLDs give noticeably different gTTS accents for the same lang.
        if c.voice_gender == "female":
            return "co.uk"
        if c.voice_age == "elderly":
            return "co.in"
        if c.role == "narrator":
            return "com"
        return "com.au"

    @staticmethod
    def _rate_for(c: Character) -> int:
        if "whisper" in c.voice_style.lower():
            return 140
        if "energetic" in c.voice_style.lower() or c.voice_age == "child":
            return 200
        return 175

    # ---- TTS rendering ---------------------------------------------------

    def _render_segments(
        self, state: PipelineState, voice_configs: List[VoiceConfig]
    ) -> tuple[List[AudioSegment], TimingManifest]:
        proj = project_dir(state.project_id)
        audio_dir = proj / "audio"
        audio_dir.mkdir(exist_ok=True, parents=True)
        cfg_by_char: Dict[str, VoiceConfig] = {v.character_id: v for v in voice_configs}

        segments: List[AudioSegment] = []
        cursor_ms = 0
        for scene in state.script.scenes:
            scene_segments: List[AudioSegment] = []
            for line in scene.dialogue:
                cfg = cfg_by_char.get(line.character_id)
                out_file = audio_dir / f"{scene.scene_id}_{line.line_id}.mp3"
                tts_res = self.tools.execute(
                    "audio.tts",
                    text=line.text,
                    out_path=str(out_file),
                    engine=cfg.engine if cfg else "gtts",
                    language=cfg.language if cfg else "en",
                    tld=cfg.tld if cfg else "com",
                    rate=cfg.rate if cfg else 175,
                )
                if not tts_res.success:
                    log.warning("tts failed for %s: %s — using silent placeholder",
                                line.line_id, tts_res.error)
                    out_file = audio_dir / f"{scene.scene_id}_{line.line_id}.wav"
                    self.tools.execute("audio.tts", text=line.text,
                                       out_path=str(out_file), engine="silent")
                rendered = Path(tts_res.data) if tts_res.success else out_file
                duration_ms = self._probe_duration_ms(rendered) or line.duration_ms
                seg = AudioSegment(
                    segment_id=f"{scene.scene_id}_{line.line_id}",
                    scene_id=scene.scene_id,
                    line_id=line.line_id,
                    character_id=line.character_id,
                    file_path=str(rendered),
                    kind="dialogue",
                    start_ms=cursor_ms,
                    end_ms=cursor_ms + duration_ms,
                    duration_ms=duration_ms,
                    text=line.text,
                )
                segments.append(seg)
                scene_segments.append(seg)
                cursor_ms += duration_ms

            # If the scene's dialogue is shorter than its declared duration_ms, pad with silence.
            scene_used = sum(s.duration_ms for s in scene_segments)
            if scene_used < scene.duration_ms:
                gap = scene.duration_ms - scene_used
                cursor_ms += gap

        total_ms = cursor_ms or sum(s.duration_ms for s in state.script.scenes)
        manifest = TimingManifest(
            project_id=state.project_id,
            total_duration_ms=total_ms,
            segments=segments,
        )
        return segments, manifest

    # ---- BGM -------------------------------------------------------------

    def _render_bgm(self, state: PipelineState) -> Optional[str]:
        proj = project_dir(state.project_id)
        bgm_dir = proj / "audio" / "bgm"
        bgm_dir.mkdir(exist_ok=True, parents=True)
        per_scene_paths: List[str] = []
        for scene in state.script.scenes:
            out_file = bgm_dir / f"{scene.scene_id}_bgm.wav"
            res = self.tools.execute(
                "audio.bgm",
                mood=scene.music_mood,
                duration_ms=scene.duration_ms,
                out_path=str(out_file),
            )
            if res.success:
                per_scene_paths.append(res.data)
        if not per_scene_paths:
            return None
        # Concatenate the per-scene BGM clips into one bed track.
        merged = bgm_dir / "bgm_full.wav"
        merge = self.tools.execute("audio.merge",
                                   segments=per_scene_paths,
                                   out_path=str(merged))
        return merge.data if merge.success else per_scene_paths[0]

    # ---- master track ----------------------------------------------------

    def _mix_master(self, state: PipelineState,
                    segments: List[AudioSegment], bgm_path: Optional[str]) -> str:
        proj = project_dir(state.project_id)
        out = proj / "audio" / "master.wav"
        seg_paths = [s.file_path for s in segments if s.kind == "dialogue"]
        merge = self.tools.execute(
            "audio.merge",
            segments=seg_paths,
            out_path=str(out),
            bgm=bgm_path,
            bgm_volume=0.18,
        )
        return merge.data if merge.success else str(out)

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _probe_duration_ms(path: Path) -> Optional[int]:
        if not path.exists():
            return None
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, check=True,
            )
            return int(float(r.stdout.strip()) * 1000)
        except Exception:  # noqa: BLE001
            return None

    # ---- serialization ---------------------------------------------------

    def _serialize(self, project_id: str, output: AudioOutput) -> List[str]:
        proj = project_dir(project_id)
        manifest_path = proj / "timing_manifest.json"
        write_json(manifest_path, output.manifest.model_dump(mode="json"))
        audio_summary = proj / "audio_summary.json"
        write_json(audio_summary, {
            "project_id": project_id,
            "phase": PHASE_AUDIO,
            "status": "complete",
            "voice_configs": [v.model_dump(mode="json") for v in output.voice_configs],
            "segment_count": len(output.manifest.segments),
            "bgm_track": output.bgm_track,
            "master_track": output.master_track,
        })
        artifacts = [str(manifest_path), str(audio_summary)]
        if output.bgm_track:
            artifacts.append(output.bgm_track)
        if output.master_track:
            artifacts.append(output.master_track)
        artifacts.extend(s.file_path for s in output.manifest.segments)
        return artifacts
