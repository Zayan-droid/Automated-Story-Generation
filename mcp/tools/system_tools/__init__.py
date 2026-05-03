"""System tools — file ops, state ops, logging."""
from .file_tool import FileReadTool, FileWriteTool, FileDeleteTool
from .state_tool import StateSnapshotTool, StateRevertTool, StateHistoryTool
from .logger_tool import LoggerTool
from mcp.tool_registry import registry

registry.register(FileReadTool())
registry.register(FileWriteTool())
registry.register(FileDeleteTool())
registry.register(StateSnapshotTool())
registry.register(StateRevertTool())
registry.register(StateHistoryTool())
registry.register(LoggerTool())

__all__ = [
    "FileReadTool", "FileWriteTool", "FileDeleteTool",
    "StateSnapshotTool", "StateRevertTool", "StateHistoryTool",
    "LoggerTool",
]
