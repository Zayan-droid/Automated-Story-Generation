"""Structured logger tool — exposes the shared logger as an MCP-style call."""
from __future__ import annotations

from mcp.base_tool import BaseTool, ToolResult
from shared.utils.logging import get_logger


class LoggerTool(BaseTool):
    name = "system.log"
    description = "Log a message at a given level."
    category = "system"

    def run(self, message: str, level: str = "info", source: str = "agent", **_) -> ToolResult:
        log = get_logger(source)
        getattr(log, level.lower(), log.info)(message)
        return ToolResult(success=True, data=message)
