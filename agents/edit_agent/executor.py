"""Edit step executor — runs the targeted re-runs implied by an EditStep."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import List

from mcp.tool_executor import ToolExecutor
from shared.schemas.audio import AudioSegment
from shared.schemas.pipeline import PipelineState
from shared.utils.files import project_dir
from shared.utils.logging import get_logger

from .planner import EditStep

log = get_logger("edit_executor")


class EditExecutor:
    """Carries out the re-runs implied by an edit step.

    Each handler returns a list of asset paths that were created/modified, so
    the caller (EditAgent) can snapshot them for undo.
    """

    def __init__(self):
        self.tools = ToolExecutor()

    # ---- entry -----------------------------------------------------------

    def execute(self, state: PipelineState, step: EditStep) -> List[str]:
        log.info("executing step %s (%s/%s) params=%s",
                 step.name, step.target, step.scope, step.params)
        handler = getattr(self, f"_step_{step.name}", None)
        if not handler:
            raise ValueError(f"no handler for step '{step.name}'")
        return handler(state, step) or []

    # ---- audio handlers --------------------------------------------------

    def _step_rerun_audio(self, state: PipelineState, step: EditStep) -> List[str]:
        from agents.audio_agent import AudioAgent
        if step.scope.startswith("character:"):
            char_id = step.scope.split(":", 1)[1]
            return self._rerun_char_audio(state, char_id, step.params)
        if step.scope.startswith("scene:"):
            scene_id = step.scope.split(":", 1)[1]
            return self._rerun_scene_audio(state, scene_id, step.params)
        # full re-run
        AudioAgent().run(state, with_bgm=bool(state.audio and state.audio.bgm_track))
        return state.phase2.artifact_paths

    def _rerun_char_audio(self, state: PipelineState, char_id: str,
                          params: dict) -> List[str]:
        if not state.audio:
            return []
        # Adjust the affected voice config.
        for v in state.audio.voice_configs:
            if v.character_id == char_id:
                tone = params.get("tone")
                if tone:
                    v.tone = tone
                    if tone == "whispered":
                        v.rate = 130
                    elif tone == "deep":
                        v.tld = "co.uk"
                    elif tone == "cheerful":
                        v.rate = 200
                break

        # Re-render the affected segments.
        proj = project_dir(state.project_id)
        audio_dir = proj / "audio"
        affected: List[str] = []
        cursor_ms = 0
        new_segments: List[AudioSegment] = []
        cfg_by_char = {v.character_id: v for v in state.audio.voice_configs}
        for seg in state.audio.manifest.segments:
            if seg.character_id == char_id and seg.kind == "dialogue":
                cfg = cfg_by_char[char_id]
                out_file = audio_dir / Path(seg.file_path).name
                self.tools.execute(
                    "audio.tts",
                    text=seg.text,
                    out_path=str(out_file),
                    engine=cfg.engine,
                    language=cfg.language,
                    tld=cfg.tld,
                    rate=cfg.rate,
                )
                affected.append(str(out_file))
                seg.file_path = str(out_file)
                duration = self._probe_ms(out_file) or seg.duration_ms
                seg.duration_ms = duration
            seg.start_ms = cursor_ms
            seg.end_ms = cursor_ms + seg.duration_ms
            cursor_ms += seg.duration_ms
            new_segments.append(seg)

        state.audio.manifest.segments = new_segments
        state.audio.manifest.total_duration_ms = cursor_ms

        # Remix the master.
        master_out = proj / "audio" / "master.wav"
        seg_paths = [s.file_path for s in new_segments if s.kind == "dialogue"]
        self.tools.execute("audio.merge",
                           segments=seg_paths,
                           out_path=str(master_out),
                           bgm=state.audio.bgm_track,
                           bgm_volume=0.18)
        state.audio.master_track = str(master_out)
        affected.append(str(master_out))
        return affected

    def _rerun_scene_audio(self, state: PipelineState, scene_id: str,
                           params: dict) -> List[str]:
        if not state.audio or not state.script:
            return []
        proj = project_dir(state.project_id)
        affected: List[str] = []
        for seg in state.audio.manifest.segments:
            if seg.scene_id != scene_id or seg.kind != "dialogue":
                continue
            cfg = next((v for v in state.audio.voice_configs
                        if v.character_id == seg.character_id), None)
            if not cfg:
                continue
            out_file = proj / "audio" / Path(seg.file_path).name
            self.tools.execute(
                "audio.tts",
                text=seg.text,
                out_path=str(out_file),
                engine=cfg.engine, language=cfg.language,
                tld=cfg.tld, rate=cfg.rate,
            )
            affected.append(str(out_file))
            seg.file_path = str(out_file)
        # Remix master.
        master_out = proj / "audio" / "master.wav"
        seg_paths = [s.file_path for s in state.audio.manifest.segments if s.kind == "dialogue"]
        self.tools.execute("audio.merge", segments=seg_paths, out_path=str(master_out),
                           bgm=state.audio.bgm_track, bgm_volume=0.18)
        state.audio.master_track = str(master_out)
        affected.append(str(master_out))
        return affected

    def _step_regenerate_bgm(self, state: PipelineState, step: EditStep) -> List[str]:
        if not state.audio or not state.script:
            return []
        proj = project_dir(state.project_id)
        bgm_dir = proj / "audio" / "bgm"
        bgm_dir.mkdir(parents=True, exist_ok=True)
        mood = step.params.get("mood", "ambient")
        scene_paths: List[str] = []
        for s in state.script.scenes:
            out = bgm_dir / f"{s.scene_id}_bgm.wav"
            res = self.tools.execute("audio.bgm", mood=mood,
                                     duration_ms=s.duration_ms, out_path=str(out))
            if res.success:
                scene_paths.append(res.data)
        merged = bgm_dir / "bgm_full.wav"
        self.tools.execute("audio.merge", segments=scene_paths, out_path=str(merged))
        state.audio.bgm_track = str(merged)
        # Remix.
        master_out = proj / "audio" / "master.wav"
        seg_paths = [s.file_path for s in state.audio.manifest.segments if s.kind == "dialogue"]
        self.tools.execute("audio.merge", segments=seg_paths, out_path=str(master_out),
                           bgm=str(merged), bgm_volume=0.18)
        state.audio.master_track = str(master_out)
        return [str(merged), str(master_out)]

    def _step_disable_bgm(self, state: PipelineState, step: EditStep) -> List[str]:
        if not state.audio:
            return []
        state.audio.bgm_track = None
        proj = project_dir(state.project_id)
        master_out = proj / "audio" / "master.wav"
        seg_paths = [s.file_path for s in state.audio.manifest.segments if s.kind == "dialogue"]
        self.tools.execute("audio.merge", segments=seg_paths, out_path=str(master_out))
        state.audio.master_track = str(master_out)
        return [str(master_out)]

    # ---- video frame handlers -------------------------------------------

    def _step_regenerate_scene(self, state: PipelineState, step: EditStep) -> List[str]:
        """Regenerate the establishing image AND re-render every shot in the scene."""
        if not state.script or not state.video:
            return []
        from agents.video_agent.animator import Shot as RenderShot, assemble_scene, render_shot
        proj = project_dir(state.project_id)
        shot_dir = proj / "video" / "shots"
        scene_dir = proj / "video" / "scenes"
        shot_dir.mkdir(parents=True, exist_ok=True)
        scene_dir.mkdir(parents=True, exist_ok=True)

        target_ids = self._scene_ids_in_scope(state, step.scope)
        affected: List[str] = []
        for scene_id in target_ids:
            scene = next((s for s in state.script.scenes if s.scene_id == scene_id), None)
            frame = next((f for f in state.video.frames if f.scene_id == scene_id), None)
            if not scene or not frame:
                continue

            # New establishing image (varying the seed by version).
            img_out = proj / "video" / "frames" / f"{scene_id}.png"
            self.tools.execute(
                "vision.generate_image",
                prompt=scene.visual_prompt,
                out_path=str(img_out),
                width=state.video.width, height=state.video.height,
                style="cinematic, dramatic lighting, highly detailed",
                seed=hash(scene_id + str(state.version)) & 0x7FFFFFFF,
            )
            frame.image_path = str(img_out)

            # Re-render each shot using its existing motion + duration.
            rendered_paths: List[Path] = []
            for shot in frame.shots:
                # Pick the source: establishing shots get the new image,
                # character/lip-sync shots keep their portrait/clip.
                src = str(img_out) if shot.kind == "establishing" else shot.image_path
                rs = RenderShot(
                    image_path=src,
                    duration_ms=shot.duration_ms,
                    motion=shot.motion,
                    audio_path=shot.audio_path,
                    is_lip_sync=False,
                )
                clip_path = shot_dir / f"{shot.shot_id}.mp4"
                render_shot(rs, clip_path,
                            state.video.width, state.video.height, state.video.fps)
                shot.clip_path = str(clip_path)
                if shot.kind == "establishing":
                    shot.image_path = str(img_out)
                rendered_paths.append(clip_path)
                affected.append(str(clip_path))

            # Re-stitch the scene composite.
            scene_clip = scene_dir / f"{scene_id}.mp4"
            assemble_scene(rendered_paths, scene_clip, crossfade_ms=200)
            frame.clip_path = str(scene_clip)
            affected.extend([str(img_out), str(scene_clip)])
        return affected

    def _step_apply_filter(self, state: PipelineState, step: EditStep) -> List[str]:
        """Apply a filter to all shots' source images in the targeted scenes,
        then re-render each shot and reassemble the scene clip."""
        if not state.video:
            return []
        from agents.video_agent.animator import Shot as RenderShot, assemble_scene, render_shot
        filt = step.params.get("filter") or step.params.get("filter_name") \
               or step.params.get("aesthetic", "darker")
        target_ids = self._scene_ids_in_scope(state, step.scope)
        proj = project_dir(state.project_id)
        shot_dir = proj / "video" / "shots"
        scene_dir = proj / "video" / "scenes"
        shot_dir.mkdir(parents=True, exist_ok=True)
        scene_dir.mkdir(parents=True, exist_ok=True)

        affected: List[str] = []
        for scene_id in target_ids:
            frame = next((f for f in state.video.frames if f.scene_id == scene_id), None)
            if not frame:
                continue
            # Apply the filter to the establishing image.
            est_in = frame.image_path
            est_out = proj / "video" / "frames" / f"{scene_id}.png"
            self.tools.execute(
                "vision.edit_image",
                in_path=est_in, out_path=str(est_out), filters=[filt],
            )
            frame.image_path = str(est_out)

            # Re-render every shot (establishing shots use the filtered image).
            rendered_paths: List[Path] = []
            for shot in frame.shots:
                if shot.kind == "establishing":
                    src = str(est_out)
                    shot.image_path = src
                else:
                    src = shot.image_path
                rs = RenderShot(
                    image_path=src,
                    duration_ms=shot.duration_ms,
                    motion=shot.motion,
                    audio_path=shot.audio_path,
                )
                clip_path = shot_dir / f"{shot.shot_id}.mp4"
                render_shot(rs, clip_path,
                            state.video.width, state.video.height, state.video.fps)
                shot.clip_path = str(clip_path)
                rendered_paths.append(clip_path)
                affected.append(str(clip_path))

            # If the scene has no shots (rare — silent scenes), do a single re-render.
            if not rendered_paths:
                rs = RenderShot(image_path=str(est_out),
                                duration_ms=frame.duration_ms,
                                motion=frame.motion or "ken_burns_diag")
                clip_path = scene_dir / f"{scene_id}.mp4"
                render_shot(rs, clip_path,
                            state.video.width, state.video.height, state.video.fps)
                frame.clip_path = str(clip_path)
                affected.append(str(clip_path))
            else:
                scene_clip = scene_dir / f"{scene_id}.mp4"
                assemble_scene(rendered_paths, scene_clip, crossfade_ms=200)
                frame.clip_path = str(scene_clip)
                affected.append(str(scene_clip))
            affected.append(str(est_out))
        return affected

    def _step_regenerate_all_scenes(self, state: PipelineState, step: EditStep) -> List[str]:
        # Mirror the regenerate_scene handler but across all scenes.
        if not state.script or not state.video:
            return []
        all_step = EditStep(name="regenerate_scene", target="video_frame",
                            scope="global", params=step.params)
        return self._step_regenerate_scene(state, all_step)

    # ---- video handlers --------------------------------------------------

    def _step_recompose_video(self, state: PipelineState, step: EditStep) -> List[str]:
        if not state.video:
            return []
        proj = project_dir(state.project_id)
        out = proj / "final_output.mp4"
        clips = [f.clip_path for f in state.video.frames if f.clip_path]
        master = state.audio.master_track if state.audio else None
        self.tools.execute("video.compose", clips=clips, out_path=str(out),
                           audio_path=master, transition="fade", transition_ms=400)

        want_subtitles = step.params.get("subtitles", state.video.has_subtitles)
        final = out
        if want_subtitles and state.audio:
            sub_lines = [
                {"start_ms": s.start_ms, "end_ms": s.end_ms, "text": s.text}
                for s in state.audio.manifest.segments
                if s.kind == "dialogue" and s.text
            ]
            if sub_lines:
                subbed = proj / "final_output_subtitled.mp4"
                self.tools.execute("video.subtitle", in_path=str(out),
                                   out_path=str(subbed), lines=sub_lines)
                final = subbed
        state.video.has_subtitles = bool(want_subtitles)
        state.video.final_video_path = str(final)
        return [str(out), str(final)]

    def _step_change_speed(self, state: PipelineState, step: EditStep) -> List[str]:
        if not state.video:
            return []
        factor = float(step.params.get("factor", 1.5))
        # ffmpeg setpts (video) + atempo (audio) chain.
        proj = project_dir(state.project_id)
        in_v = state.video.final_video_path
        out = proj / "final_output_speed.mp4"
        # atempo is bounded [0.5, 100]; chain if needed.
        atempo = factor
        af_chain = []
        while atempo > 2.0:
            af_chain.append("atempo=2.0")
            atempo /= 2.0
        while atempo < 0.5:
            af_chain.append("atempo=0.5")
            atempo *= 2.0
        af_chain.append(f"atempo={atempo:.4f}")
        af = ",".join(af_chain)
        cmd = [
            "ffmpeg", "-y", "-i", in_v,
            "-filter_complex",
            f"[0:v]setpts={1/factor:.4f}*PTS[v];[0:a]{af}[a]",
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode == 0:
            state.video.final_video_path = str(out)
            return [str(out)]
        return []

    # ---- script handler --------------------------------------------------

    def _step_regenerate_script(self, state: PipelineState, step: EditStep) -> List[str]:
        from agents.story_agent import StoryAgent
        # Optionally bias the prompt with a new genre/style.
        params = step.params or {}
        if params.get("genre"):
            state.user_prompt = f"{state.user_prompt} (genre: {params['genre']})"
        StoryAgent().run(state, target_duration_s=state.script.story.target_duration_s,
                         scene_count=len(state.script.scenes))
        return state.phase1.artifact_paths

    def _step_rerun_video(self, state: PipelineState, step: EditStep) -> List[str]:
        from agents.video_agent import VideoAgent
        VideoAgent().run(state, with_subtitles=state.video.has_subtitles if state.video else True,
                         width=state.video.width if state.video else 1280,
                         height=state.video.height if state.video else 720,
                         fps=state.video.fps if state.video else 24)
        return state.phase3.artifact_paths

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def _scene_ids_in_scope(state: PipelineState, scope: str) -> List[str]:
        if scope.startswith("scene:"):
            return [scope.split(":", 1)[1]]
        if state.script:
            return [s.scene_id for s in state.script.scenes]
        return []

    @staticmethod
    def _probe_ms(path: Path) -> int | None:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, check=True,
            )
            return int(float(r.stdout.strip()) * 1000)
        except Exception:  # noqa: BLE001
            return None
