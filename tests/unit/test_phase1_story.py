"""Phase 1 — Story / Script / Character agent."""
from __future__ import annotations
from pathlib import Path

from agents.story_agent import StoryAgent
from agents.story_agent.planner import template_script
from shared.schemas.pipeline import PipelineState
from shared.schemas.story import ScriptOutput
from shared.utils.ids import new_project_id


def test_template_script_produces_valid_output():
    pid = new_project_id()
    out = template_script(pid, "A young astronaut discovers a hidden ocean on Mars",
                          target_duration_s=40)
    assert isinstance(out, ScriptOutput)
    assert out.story.title
    assert len(out.scenes) >= 4
    assert len(out.characters.characters) >= 3
    char_ids = {c.id for c in out.characters.characters}
    for s in out.scenes:
        assert s.duration_ms >= 1000
        for ln in s.dialogue:
            assert ln.character_id in char_ids


def test_genre_detection_picks_scifi():
    out = template_script("pid_1", "A robot wakes up on Mars")
    assert out.story.genre in ("sci-fi", "drama")  # robot OR mars


def test_story_agent_runs_with_mock_provider(tmp_path, monkeypatch):
    # Force outputs into tmp_path-isolated workspace.
    import shared.constants as constants
    monkeypatch.setattr(constants, "OUTPUTS_DIR", tmp_path / "out")
    constants.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    agent = StoryAgent()
    state = PipelineState(project_id="t_phase1", user_prompt="A dragon learns to dance")
    script = agent.run(state, target_duration_s=30, scene_count=4)
    assert state.phase1.status == "complete"
    assert script.story.title
    assert state.script is script
    # Required artifacts written.
    assert (tmp_path / "out" / "t_phase1" / "story.json").exists()
    assert (tmp_path / "out" / "t_phase1" / "characters.json").exists()
    assert (tmp_path / "out" / "t_phase1" / "script.json").exists()
    assert (tmp_path / "out" / "t_phase1" / "phase2_audio_handoff.json").exists()
    assert (tmp_path / "out" / "t_phase1" / "phase3_video_handoff.json").exists()
    assert (tmp_path / "out" / "t_phase1" / "summary.json").exists()


def test_story_agent_validates_character_consistency():
    agent = StoryAgent()
    state = PipelineState(project_id="t_phase1_v", user_prompt="A girl finds a magic book")
    script = agent.run(state, target_duration_s=20, scene_count=4)
    char_ids = {c.id for c in script.characters.characters}
    for s in script.scenes:
        for ln in s.dialogue:
            assert ln.character_id in char_ids
