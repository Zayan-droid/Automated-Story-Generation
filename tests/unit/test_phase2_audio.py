"""Phase 2 — Audio generation tests (use silent TTS to stay offline)."""
from __future__ import annotations
from pathlib import Path

from agents.audio_agent import AudioAgent
from agents.story_agent.planner import template_script
from mcp.tool_executor import ToolExecutor
from shared.schemas.pipeline import PipelineState


def test_silent_tts_creates_wav(tmp_path):
    out = tmp_path / "silent.wav"
    res = ToolExecutor().execute("audio.tts", text="hello world",
                                 out_path=str(out), engine="silent")
    assert res.success, res.error
    assert Path(res.data).exists()


def test_bgm_tool_creates_audio(tmp_path):
    out = tmp_path / "bgm.wav"
    res = ToolExecutor().execute("audio.bgm", mood="ambient",
                                 duration_ms=2000, out_path=str(out))
    assert res.success, res.error
    assert Path(res.data).exists()


def test_audio_merger(tmp_path):
    seg_a = tmp_path / "a.wav"
    seg_b = tmp_path / "b.wav"
    ToolExecutor().execute("audio.tts", text="alpha", out_path=str(seg_a),
                           engine="silent")
    ToolExecutor().execute("audio.tts", text="beta", out_path=str(seg_b),
                           engine="silent")
    out = tmp_path / "merged.wav"
    res = ToolExecutor().execute("audio.merge",
                                 segments=[str(seg_a), str(seg_b)],
                                 out_path=str(out))
    assert res.success, res.error
    assert Path(res.data).exists()


def test_audio_agent_full_run(tmp_path, monkeypatch):
    import shared.constants as constants
    monkeypatch.setattr(constants, "OUTPUTS_DIR", tmp_path / "out")
    constants.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    state = PipelineState(project_id="t_phase2", user_prompt="A robot finds love")
    state.script = template_script("t_phase2", state.user_prompt, target_duration_s=20)

    # Force the silent TTS path so the test is fully offline.
    agent = AudioAgent()
    # Monkeypatch the TTS tool to always use silent engine.
    real_execute = agent.tools.execute
    def patched(tool, **kwargs):
        if tool == "audio.tts":
            kwargs["engine"] = "silent"
        return real_execute(tool, **kwargs)
    agent.tools.execute = patched

    out = agent.run(state, with_bgm=False)
    assert state.phase2.status == "complete"
    assert out.manifest.segments
    assert (tmp_path / "out" / "t_phase2" / "timing_manifest.json").exists()
    for seg in out.manifest.segments:
        assert Path(seg.file_path).exists()
