"""State manager — append-only versioning + snapshot + revert."""
from __future__ import annotations
from pathlib import Path

from shared.schemas.pipeline import PipelineState
from state_manager.storage import SqliteStorage
from state_manager.state_manager import StateManager


def test_snapshot_assigns_increasing_versions(tmp_path, monkeypatch):
    import shared.constants as constants
    monkeypatch.setattr(constants, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(constants, "OUTPUTS_DIR", tmp_path / "out")
    monkeypatch.setattr(constants, "DB_PATH", tmp_path / "state.db")
    constants.STATE_DIR.mkdir(parents=True, exist_ok=True)
    constants.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    sm = StateManager(SqliteStorage(tmp_path / "state.db"))
    s1 = PipelineState(project_id="p1", user_prompt="x")
    v1 = sm.snapshot(s1, asset_paths=[], description="initial")
    v2 = sm.snapshot(s1, asset_paths=[], description="edit_1")
    v3 = sm.snapshot(s1, asset_paths=[], description="edit_2")
    assert v1.version == 1 and v2.version == 2 and v3.version == 3
    history = sm.history("p1")
    assert [h["version"] for h in history] == [1, 2, 3]


def test_snapshot_persists_assets(tmp_path, monkeypatch):
    import shared.constants as constants
    monkeypatch.setattr(constants, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(constants, "OUTPUTS_DIR", tmp_path / "out")
    monkeypatch.setattr(constants, "DB_PATH", tmp_path / "db.sqlite")
    constants.STATE_DIR.mkdir(parents=True, exist_ok=True)
    constants.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    proj_root = constants.OUTPUTS_DIR / "p2"
    proj_root.mkdir()
    asset = proj_root / "data.txt"
    asset.write_text("hello v1", encoding="utf-8")

    sm = StateManager(SqliteStorage(tmp_path / "db.sqlite"))
    s = PipelineState(project_id="p2", user_prompt="x")
    sm.snapshot(s, asset_paths=[str(asset)], description="v1")

    asset.write_text("hello v2", encoding="utf-8")
    sm.snapshot(s, asset_paths=[str(asset)], description="v2")

    # Revert to v1 should restore the original content.
    sm.revert("p2", 1)
    assert asset.read_text(encoding="utf-8") == "hello v1"


def test_edit_log_round_trip(tmp_path, monkeypatch):
    import shared.constants as constants
    monkeypatch.setattr(constants, "DB_PATH", tmp_path / "db.sqlite")
    sm = StateManager(SqliteStorage(tmp_path / "db.sqlite"))
    sm.log_edit("p3", "make it darker",
                {"intent": "adjust_scene_aesthetic", "target": "video_frame",
                 "scope": "global", "parameters": {}, "confidence": 0.8,
                 "reasoning": "test"},
                {"success": True})
    log = sm.edit_history("p3")
    assert len(log) == 1
    assert log[0]["query"] == "make it darker"
    assert log[0]["intent"]["target"] == "video_frame"
