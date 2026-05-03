"""pytest configuration: ensure project root is on sys.path."""
import os
import sys
from pathlib import Path

# Project root.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force mock LLM provider for deterministic tests.
os.environ.setdefault("LLM_PROVIDER", "mock")
# Disable Pollinations.ai network call by default in tests so they're offline-safe.
os.environ.setdefault("POLLINATIONS_DISABLE", "1")

# Register all MCP tools.
import mcp.tools  # noqa: F401, E402
