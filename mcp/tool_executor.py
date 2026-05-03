"""Lookup + invoke tools by name."""
from __future__ import annotations
from typing import Dict

from .base_tool import ToolResult
from .tool_registry import ToolRegistry, registry


class ToolExecutor:
    def __init__(self, reg: ToolRegistry = registry):
        self.registry = reg

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        tool = self.registry.get(tool_name)
        if not tool:
            return ToolResult(success=False, error=f"tool '{tool_name}' not registered")
        return tool.safe_run(**kwargs)

    def list_tools(self) -> Dict[str, str]:
        return {t.name: t.description for t in self.registry.all()}
