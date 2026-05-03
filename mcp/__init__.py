"""MCP layer — tool abstraction so agents are decoupled from concrete implementations."""
from .base_tool import BaseTool, ToolResult
from .tool_registry import ToolRegistry, registry
from .tool_executor import ToolExecutor

__all__ = ["BaseTool", "ToolResult", "ToolRegistry", "registry", "ToolExecutor"]
