"""Base interface every MCP tool implements."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolResult:
    """Uniform return type from every tool."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """All concrete tools subclass this and implement run()."""

    name: str = ""
    description: str = ""
    category: str = "generic"

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool with keyword arguments."""

    def safe_run(self, **kwargs) -> ToolResult:
        try:
            return self.run(**kwargs)
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, error=f"{type(e).__name__}: {e}")
