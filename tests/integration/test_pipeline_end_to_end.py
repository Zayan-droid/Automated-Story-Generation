"""End-to-end smoke test: prompt -> playable MP4 in mock/template mode."""
from __future__ import annotations
from pathlib import Path

import pytest

from agents.orchestrator import PipelineOrchestrator
from shared.utils.ids import new_project_id


@pytest.mark.timeout(180)
def test_full_pipeline_produces_video(tmp_path, monkeypatch):
    # Isolate outputs.
    import shared.constants as constants
    monkeypatch.setattr(constants, "OUTPUTS_DIR", tmp_path / "out")
    monkeypatch.setattr(constants, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(constants, "DB_PATH", tmp_path / "state.db")
    constants.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    constants.STATE_DIR.mkdir(parents=True, exist_ok=True)

    # Reset orchestrator + dependencies after monkey-patching constants.
    from state_manager.state_manager import StateManager
    from state_manager.storage import SqliteStorage
    sm = StateManager(SqliteStorage(tmp_path / "state.db"))

    orch = PipelineOrchestrator(state_manager=sm)
    # Patch the audio agent's TTS to use silent so we don't need internet.
    real_execute = orch.audio.tools.execute
    def patched(tool, **kwargs):
        if tool == "audio.tts":
            kwargs["engine"] = "silent"
        return real_execute(tool, **kwargs)
    orch.audio.tools.execute = patched

    project_id = new_project_id()
    state = orch.run_full(
        prompt="A robot wakes up to find the world has changed forever",
        project_id=project_id,
        target_duration_s=16,
        scene_count=4,
        with_bgm=False,
        with_subtitles=False,
    )
    assert state.video is not None
    assert Path(state.video.final_video_path).exists()
    # Video should be non-trivial in size (> 1KB).
    assert Path(state.video.final_video_path).stat().st_size > 1024
    # State manager records v1 = initial run.
    assert sm.history(project_id)[-1]["version"] >= 1
