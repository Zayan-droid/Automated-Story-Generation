"""Phase 3 — Video generation tests (use PIL placeholder + silent audio)."""
from __future__ import annotations
from pathlib import Path

from agents.audio_agent import AudioAgent
from agents.story_agent.planner import template_script
from agents.video_agent import VideoAgent
from mcp.tool_executor import ToolExecutor
from shared.schemas.pipeline import PipelineState


def test_image_gen_pil_fallback(tmp_path):
    out = tmp_path / "frame.png"
    res = ToolExecutor().execute(
        "vision.generate_image",
        prompt="a quiet forest at dusk",
        out_path=str(out), width=320, height=180,
    )
    assert res.success
    assert Path(res.data).exists()


def test_image_to_clip(tmp_path):
    img = tmp_path / "img.png"
    ToolExecutor().execute("vision.generate_image", prompt="a wide ocean",
                           out_path=str(img), width=320, height=180)
    out = tmp_path / "clip.mp4"
    res = ToolExecutor().execute(
        "video.image_to_clip",
        image_path=str(img), out_path=str(out),
        duration_ms=1500, width=320, height=180, fps=12, motion="ken_burns",
    )
    assert res.success, res.error
    assert Path(res.data).exists()


def test_video_compose(tmp_path):
    clips = []
    for i in range(2):
        img = tmp_path / f"i_{i}.png"
        clip = tmp_path / f"c_{i}.mp4"
        ToolExecutor().execute("vision.generate_image", prompt=f"scene {i}",
                               out_path=str(img), width=320, height=180)
        ToolExecutor().execute(
            "video.image_to_clip",
            image_path=str(img), out_path=str(clip),
            duration_ms=1200, width=320, height=180, fps=12, motion="none",
        )
        clips.append(str(clip))
    out = tmp_path / "final.mp4"
    res = ToolExecutor().execute("video.compose", clips=clips,
                                 out_path=str(out), transition="fade")
    assert res.success, res.error
    assert Path(res.data).exists()


def test_video_agent_full_run(tmp_path, monkeypatch):
    import shared.constants as constants
    monkeypatch.setattr(constants, "OUTPUTS_DIR", tmp_path / "out")
    constants.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    state = PipelineState(project_id="t_phase3", user_prompt="A whale sings")
    state.script = template_script("t_phase3", state.user_prompt, target_duration_s=16)

    audio = AudioAgent()
    # Patch TTS to silent so the audio phase is fast/offline.
    real_execute = audio.tools.execute
    def patched(tool, **kwargs):
        if tool == "audio.tts":
            kwargs["engine"] = "silent"
        return real_execute(tool, **kwargs)
    audio.tools.execute = patched
    audio.run(state, with_bgm=False)

    video = VideoAgent()
    out = video.run(state, with_subtitles=False, width=320, height=180, fps=12)
    assert state.phase3.status == "complete"
    assert Path(out.final_video_path).exists()
    assert len(out.frames) == len(state.script.scenes)
