"""StateManager bindings as MCP tools."""
from __future__ import annotations
from typing import List

from mcp.base_tool import BaseTool, ToolResult
from state_manager.state_manager import StateManager
from state_manager.history import format_history
from shared.schemas.pipeline import PipelineState


_sm = StateManager()


class StateSnapshotTool(BaseTool):
    name = "system.state_snapshot"
    description = "Snapshot the pipeline state and assets at the current moment."
    category = "system"

    def run(self, state: dict, asset_paths: List[str], description: str = "",
            edit_intent: dict | None = None, **_) -> ToolResult:
        ps = PipelineState.model_validate(state)
        version = _sm.snapshot(ps, asset_paths or [], description=description,
                               edit_intent=edit_intent)
        return ToolResult(success=True, data=version.model_dump(mode="json"))


class StateRevertTool(BaseTool):
    name = "system.state_revert"
    description = "Revert a project to a previous version."
    category = "system"

    def run(self, project_id: str, version: int, **_) -> ToolResult:
        st = _sm.revert(project_id, version)
        return ToolResult(success=True, data=st.model_dump(mode="json"))


class StateHistoryTool(BaseTool):
    name = "system.state_history"
    description = "Return the version history for a project."
    category = "system"

    def run(self, project_id: str, **_) -> ToolResult:
        rows = _sm.history(project_id)
        return ToolResult(success=True, data=format_history(rows))
