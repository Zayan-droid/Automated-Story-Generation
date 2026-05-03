"""Phase 3 agent: ScriptOutput + AudioOutput -> final_output.mp4 (cinematic).

Two-tier rendering:

1. **Default (no API key required) — multi-shot ffmpeg composition.**
   - One establishing image per scene.
   - One portrait image per character (generated once, cached).
   - For each dialogue line, render a sub-clip whose duration matches the
     audio segment, using the appropriate shot (narrator -> establishing,
     character -> their portrait), with cinematic ken-burns motion.
   - Within a scene, sub-clips are crossfaded together; between scenes,
     a longer crossfade.
   - Optional vignette + film-grain post for cinematic feel.

2. **Optional upgrades when keys are configured.**
   - `FAL_KEY` / `REPLICATE_API_TOKEN` -> real text-to-video for
     establishing shots (Stable Video Diffusion / fast-SVD).
   - Same keys -> real lip-sync (SadTalker / sync-lipsync) for character
     dialogue lines, replacing the heuristic talking head.
"""
from __future__ import annotations
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from mcp.tool_executor import ToolExecutor
from shared.constants import DEFAULT_FPS, DEFAULT_HEIGHT, DEFAULT_WIDTH, PHASE_VIDEO
from shared.schemas.audio import AudioSegment
from shared.schemas.pipeline import PipelineState
from shared.schemas.video import CharacterPortrait, SceneFrame, Shot, VideoOutput
from shared.utils.files import project_dir, write_json
from shared.utils.logging import get_logger

from .animator import (
    Shot as RenderShot, assemble_scene, pick_motion_for_index, render_shot,
)

log = get_logger("video_agent")


class VideoAgent:
    def __init__(self):
        self.tools = ToolExecutor()

    # ---- public ----------------------------------------------------------

    def run(self, state: PipelineState, with_subtitles: bool = True,
            width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
            fps: int = DEFAULT_FPS,
            use_text_to_video: Optional[bool] = None,
            use_lip_sync: Optional[bool] = None,
            cinematic_post: bool = True) -> VideoOutput:
        if not state.script:
            raise ValueError("phase 3 requires state.script (run phase 1 first)")
        log.info("phase 3 start (project=%s, %dx%d@%d, subs=%s)",
                 state.project_id, width, height, fps, with_subtitles)
        state.phase3.status = "running"
        state.phase3.started_at = datetime.utcnow().isoformat()

        # Auto-detect upgrades from env if caller didn't specify.
        if use_text_to_video is None:
            use_text_to_video = bool(
                os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY") or
                os.getenv("REPLICATE_API_TOKEN") or os.getenv("HF_TOKEN")
            )
        if use_lip_sync is None:
            use_lip_sync = bool(
                os.getenv("FAL_KEY") or os.getenv("FAL_API_KEY") or
                os.getenv("REPLICATE_API_TOKEN")
            )

        try:
            # 1. Per-character portraits (used as talking-head shots).
            portraits = self._generate_portraits(state, width, height)

            # 2. Per-scene establishing image (and optional T2V clip).
            scene_assets = self._generate_scene_assets(
                state, width, height, fps, use_text_to_video,
            )

            # 3. Per-scene multi-shot composition driven by dialogue timing.
            frames = self._compose_scenes(
                state, scene_assets, portraits, width, height, fps,
                use_lip_sync=use_lip_sync, cinematic_post=cinematic_post,
            )

            # 4. Final crossfade compositor across scenes + master audio.
            final_video = self._compose_final(state, frames, with_subtitles,
                                              width, height, fps)

            output = VideoOutput(
                project_id=state.project_id,
                frames=frames,
                final_video_path=str(final_video),
                width=width, height=height, fps=fps,
                has_subtitles=with_subtitles,
                duration_ms=sum(f.duration_ms for f in frames),
                portraits=list(portraits.values()),
                used_text_to_video=use_text_to_video,
                used_lip_sync=use_lip_sync,
            )
            artifacts = self._serialize(state.project_id, output)
            state.video = output
            state.phase3.status = "complete"
            state.phase3.finished_at = datetime.utcnow().isoformat()
            state.phase3.artifact_paths = artifacts
            log.info("phase 3 complete (%d scenes, %d shots total, video=%s)",
                     len(frames), sum(len(f.shots) for f in frames), final_video)
            return output
        except Exception as e:  # noqa: BLE001
            state.phase3.status = "failed"
            state.phase3.error = f"{type(e).__name__}: {e}"
            log.exception("phase 3 failed")
            raise

    # ---- step 1: character portraits ------------------------------------

    def _generate_portraits(self, state: PipelineState, width: int, height: int
                            ) -> Dict[str, CharacterPortrait]:
        proj = project_dir(state.project_id)
        port_dir = proj / "video" / "portraits"
        port_dir.mkdir(parents=True, exist_ok=True)
        portraits: Dict[str, CharacterPortrait] = {}
        for c in state.script.characters.characters:
            out = port_dir / f"{c.id}.png"
            prompt = (
                f"close-up portrait of {c.name}, {c.visual_description}, "
                f"{c.role}, expressive face, looking at camera, "
                f"shallow depth of field, cinematic 50mm lens, "
                f"soft rim lighting, photorealistic, highly detailed"
            )
            self.tools.execute(
                "vision.generate_image",
                prompt=prompt,
                out_path=str(out),
                width=width, height=height,
                style="cinematic portrait, dramatic lighting, 8k, photorealistic",
                negative_prompt="blurry, low quality, distorted, deformed, text, watermark",
            )
            portraits[c.id] = CharacterPortrait(
                character_id=c.id,
                image_path=str(out),
            )
            log.info("  portrait: %s -> %s", c.name, out.name)
        return portraits

    # ---- step 2: scene establishing assets ------------------------------

    def _generate_scene_assets(self, state: PipelineState,
                               width: int, height: int, fps: int,
                               use_text_to_video: bool) -> Dict[str, dict]:
        proj = project_dir(state.project_id)
        img_dir = proj / "video" / "frames"
        clip_dir = proj / "video" / "clips"
        t2v_dir = proj / "video" / "t2v"
        for d in (img_dir, clip_dir, t2v_dir):
            d.mkdir(parents=True, exist_ok=True)

        assets: Dict[str, dict] = {}
        for scene in state.script.scenes:
            img_out = img_dir / f"{scene.scene_id}.png"
            self.tools.execute(
                "vision.generate_image",
                prompt=scene.visual_prompt,
                out_path=str(img_out),
                width=width, height=height,
                style="cinematic, dramatic lighting, highly detailed, 8k",
                negative_prompt="blurry, low quality, text, watermark, distorted",
            )
            entry = {"image": str(img_out), "t2v": None}

            if use_text_to_video:
                t2v_out = t2v_dir / f"{scene.scene_id}.mp4"
                # Image-to-video — pass the still we just generated to give
                # the model a strong visual anchor (it's much more reliable
                # than pure text-to-video for free-tier providers).
                res = self.tools.execute(
                    "vision.text_to_video",
                    prompt=scene.visual_prompt,
                    image_path=str(img_out),
                    out_path=str(t2v_out),
                    duration_s=min(4.0, scene.duration_ms / 1000.0),
                    width=width, height=height, fps=fps,
                )
                if res.success:
                    entry["t2v"] = res.data
                    log.info("  scene %s: t2v clip generated (%s)",
                             scene.scene_id, res.metadata.get("provider"))
                else:
                    log.info("  scene %s: t2v unavailable (%s) — using still",
                             scene.scene_id, res.error)
            assets[scene.scene_id] = entry
        return assets

    # ---- step 3: multi-shot scene composition ---------------------------

    def _compose_scenes(self, state: PipelineState, scene_assets: Dict[str, dict],
                        portraits: Dict[str, CharacterPortrait],
                        width: int, height: int, fps: int,
                        use_lip_sync: bool, cinematic_post: bool) -> List[SceneFrame]:
        proj = project_dir(state.project_id)
        shot_dir = proj / "video" / "shots"
        scene_dir = proj / "video" / "scenes"
        shot_dir.mkdir(parents=True, exist_ok=True)
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Build a scene_id -> [audio segments] map.
        seg_by_scene: Dict[str, List[AudioSegment]] = {}
        if state.audio:
            for seg in state.audio.manifest.segments:
                if seg.kind == "dialogue":
                    seg_by_scene.setdefault(seg.scene_id, []).append(seg)

        frames: List[SceneFrame] = []
        for scene in state.script.scenes:
            asset = scene_assets[scene.scene_id]
            establishing = asset["t2v"] or asset["image"]   # prefer real video
            establishing_is_video = bool(asset["t2v"])

            scene_segments = seg_by_scene.get(scene.scene_id, [])
            shots: List[Shot] = []
            rendered_paths: List[Path] = []

            if not scene_segments:
                # No dialogue -> single establishing shot for the scene's full duration.
                rs = RenderShot(
                    image_path=establishing,
                    duration_ms=scene.duration_ms,
                    motion=pick_motion_for_index(scene.index, 0),
                    is_lip_sync=establishing_is_video,
                )
                rendered = render_shot(
                    rs, shot_dir / f"{scene.scene_id}_s0.mp4",
                    width, height, fps,
                    add_grain=cinematic_post, add_vignette=cinematic_post,
                )
                rendered_paths.append(rendered)
                shots.append(Shot(
                    shot_id=f"{scene.scene_id}_s0",
                    scene_id=scene.scene_id,
                    kind="establishing",
                    image_path=asset["image"],
                    clip_path=str(rendered),
                    duration_ms=scene.duration_ms,
                    motion=rs.motion,
                ))
            else:
                # Always start with a brief establishing shot (15-30% of scene duration)
                # so the audience grasps the setting before cuts begin.
                est_ms = max(800, int(scene.duration_ms * 0.20))
                est_motion = pick_motion_for_index(scene.index, 0)
                est_rs = RenderShot(
                    image_path=establishing,
                    duration_ms=est_ms,
                    motion=est_motion,
                    is_lip_sync=establishing_is_video,
                )
                est_path = render_shot(
                    est_rs, shot_dir / f"{scene.scene_id}_est.mp4",
                    width, height, fps,
                    add_grain=cinematic_post, add_vignette=cinematic_post,
                )
                rendered_paths.append(est_path)
                shots.append(Shot(
                    shot_id=f"{scene.scene_id}_est",
                    scene_id=scene.scene_id,
                    kind="establishing",
                    image_path=asset["image"],
                    clip_path=str(est_path),
                    duration_ms=est_ms,
                    motion=est_motion,
                ))

                # One sub-clip per dialogue line, using the appropriate portrait.
                for i, seg in enumerate(scene_segments):
                    char = next(
                        (c for c in state.script.characters.characters
                         if c.id == seg.character_id), None,
                    )
                    is_narrator = char is not None and char.role == "narrator"
                    if is_narrator:
                        # Narrator lines stay on the establishing shot — different motion.
                        src = establishing
                        kind: str = "establishing"
                        char_id = None
                        is_video = establishing_is_video
                    else:
                        portrait = portraits.get(seg.character_id)
                        src = portrait.image_path if portrait else establishing
                        kind = "character"
                        char_id = seg.character_id
                        is_video = False

                    motion = pick_motion_for_index(scene.index, i + 1)
                    shot_id = f"{scene.scene_id}_l{i+1}"
                    out_path = shot_dir / f"{shot_id}.mp4"

                    # Real lip sync? Try only for character portraits (not narrator),
                    # only if we have audio, and only if the user opted in.
                    used_lip = False
                    if (use_lip_sync and not is_narrator
                            and seg.file_path and Path(seg.file_path).exists()):
                        ls_res = self.tools.execute(
                            "vision.lip_sync",
                            image_path=src,
                            audio_path=seg.file_path,
                            out_path=str(out_path),
                            duration_s=seg.duration_ms / 1000.0,
                            width=width, height=height, fps=fps,
                        )
                        if ls_res.success and ls_res.metadata.get("provider", "").startswith(("fal", "replicate")):
                            used_lip = True
                            kind = "lip_sync"
                            rendered_paths.append(out_path)
                            shots.append(Shot(
                                shot_id=shot_id, scene_id=scene.scene_id, kind=kind,
                                character_id=char_id,
                                image_path=src,
                                clip_path=str(out_path),
                                duration_ms=seg.duration_ms,
                                motion="lip_sync",
                                audio_path=seg.file_path,
                            ))
                            continue

                    # Default path: cinematic still-with-motion sub-clip.
                    rs = RenderShot(
                        image_path=src,
                        duration_ms=seg.duration_ms,
                        motion=motion,
                        is_lip_sync=is_video,
                    )
                    rendered = render_shot(
                        rs, out_path, width, height, fps,
                        add_grain=cinematic_post,
                        add_vignette=cinematic_post and is_narrator,
                    )
                    rendered_paths.append(rendered)
                    shots.append(Shot(
                        shot_id=shot_id, scene_id=scene.scene_id, kind=kind,
                        character_id=char_id,
                        image_path=src,
                        clip_path=str(rendered),
                        duration_ms=seg.duration_ms,
                        motion=motion,
                    ))

            # Stitch the scene's sub-clips together with short crossfades.
            scene_clip = scene_dir / f"{scene.scene_id}.mp4"
            assemble_scene(
                rendered_paths, scene_clip,
                crossfade_ms=200, audio_path=None,  # audio is muxed at the final stage
            )
            frames.append(SceneFrame(
                scene_id=scene.scene_id,
                image_path=asset["image"],
                clip_path=str(scene_clip),
                width=width, height=height,
                duration_ms=sum(s.duration_ms for s in shots),
                motion="multi_shot",
                transition_in=scene.transition_in,
                shots=shots,
            ))
            log.info("  scene %s composed (%d shots)", scene.scene_id, len(shots))
        return frames

    # ---- step 4: final composition --------------------------------------

    def _compose_final(self, state: PipelineState, frames: List[SceneFrame],
                       with_subtitles: bool, width: int, height: int, fps: int) -> Path:
        proj = project_dir(state.project_id)
        out = proj / "final_output.mp4"
        clips = [f.clip_path for f in frames if f.clip_path]
        master = state.audio.master_track if state.audio else None

        compose = self.tools.execute(
            "video.compose",
            clips=clips, out_path=str(out),
            audio_path=master,
            transition="fade", transition_ms=400,
        )
        if not compose.success:
            log.error("compose failed: %s", compose.error)
            return out
        if not with_subtitles or not state.audio:
            return out

        # Burn subtitles using the timing manifest.
        sub_lines = []
        for seg in state.audio.manifest.segments:
            if seg.kind == "dialogue" and seg.text:
                sub_lines.append({
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "text": seg.text,
                })
        if not sub_lines:
            return out
        subbed = proj / "final_output_subtitled.mp4"
        sub_res = self.tools.execute(
            "video.subtitle",
            in_path=str(out), out_path=str(subbed),
            lines=sub_lines, font_size=28,
        )
        if sub_res.success:
            return subbed
        log.warning("subtitle burn failed: %s", sub_res.error)
        return out

    # ---- serialization ---------------------------------------------------

    def _serialize(self, project_id: str, output: VideoOutput) -> List[str]:
        proj = project_dir(project_id)
        summary = proj / "video_summary.json"
        write_json(summary, {
            "project_id": project_id,
            "phase": PHASE_VIDEO,
            "status": "complete",
            "final_video": output.final_video_path,
            "frame_count": len(output.frames),
            "shot_count": sum(len(f.shots) for f in output.frames),
            "has_subtitles": output.has_subtitles,
            "used_text_to_video": output.used_text_to_video,
            "used_lip_sync": output.used_lip_sync,
            "width": output.width, "height": output.height, "fps": output.fps,
            "duration_ms": output.duration_ms,
            "frames": [f.model_dump(mode="json") for f in output.frames],
            "portraits": [p.model_dump(mode="json") for p in output.portraits],
        })
        artifacts: List[str] = [str(summary), output.final_video_path]
        for f in output.frames:
            artifacts.append(f.image_path)
            if f.clip_path:
                artifacts.append(f.clip_path)
            for s in f.shots:
                if s.clip_path:
                    artifacts.append(s.clip_path)
        for p in output.portraits:
            artifacts.append(p.image_path)
            if p.talking_head_clip:
                artifacts.append(p.talking_head_clip)
        return artifacts
