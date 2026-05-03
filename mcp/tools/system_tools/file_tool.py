"""Filesystem operations exposed via MCP."""
from __future__ import annotations
import shutil
from pathlib import Path

from mcp.base_tool import BaseTool, ToolResult


class FileReadTool(BaseTool):
    name = "system.file_read"
    description = "Read a text file from disk."
    category = "system"

    def run(self, path: str, **_) -> ToolResult:
        p = Path(path)
        if not p.exists():
            return ToolResult(success=False, error=f"missing: {path}")
        return ToolResult(success=True, data=p.read_text(encoding="utf-8"))


class FileWriteTool(BaseTool):
    name = "system.file_write"
    description = "Write text to a file (creates parent directories)."
    category = "system"

    def run(self, path: str, content: str, **_) -> ToolResult:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return ToolResult(success=True, data=str(p))


class FileDeleteTool(BaseTool):
    name = "system.file_delete"
    description = "Delete a file or directory."
    category = "system"

    def run(self, path: str, **_) -> ToolResult:
        p = Path(path)
        if not p.exists():
            return ToolResult(success=True, data="not present")
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return ToolResult(success=True, data=str(p))
