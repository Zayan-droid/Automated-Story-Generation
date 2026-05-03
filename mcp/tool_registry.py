"""Singleton registry of all available tools."""
from __future__ import annotations
from typing import Dict, List, Optional

from .base_tool import BaseTool


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError(f"tool {tool.__class__.__name__} has no name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def by_category(self, category: str) -> List[BaseTool]:
        return [t for t in self._tools.values() if t.category == category]

    def all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def clear(self) -> None:
        self._tools.clear()


registry = ToolRegistry()
