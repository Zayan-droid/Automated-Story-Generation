"""FastAPI entry point.

Mounts:
  /api/pipeline   — start runs, re-run phases, fetch state
  /api/edit       — natural-language edits (Phase 5)
  /api/history    — version history + revert
  /api/projects   — list known projects
  /ws/progress    — live progress events for an in-flight run
  /assets/...     — static asset server for generated images/videos
  /              — single-page HTML UI
"""
from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Ensure all MCP tools are registered.
import mcp.tools  # noqa: F401

from shared.constants import OUTPUTS_DIR

from .routes import edit as edit_routes
from .routes import history as history_routes
from .routes import pipeline as pipeline_routes
from .routes import projects as project_routes
from .websocket import progress as progress_ws


app = FastAPI(
    title="Agentic Animated Video Generation",
    version="1.0.0",
    description="End-to-end agentic pipeline: prompt → animated short film with intelligent edits.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(pipeline_routes.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(edit_routes.router, prefix="/api/edit", tags=["edit"])
app.include_router(history_routes.router, prefix="/api/history", tags=["history"])
app.include_router(project_routes.router, prefix="/api/projects", tags=["projects"])
app.include_router(progress_ws.router, prefix="/ws", tags=["websocket"])

# Static asset server — exposes the per-project output directory so the UI can
# fetch /assets/<project_id>/final_output.mp4, frame images, etc.
app.mount("/assets", StaticFiles(directory=str(OUTPUTS_DIR)), name="assets")

# Frontend — single-page app served from /frontend.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if (FRONTEND_DIR / "src").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "src")),
              name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    idx = FRONTEND_DIR / "src" / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return HTMLResponse("<h1>Agentic Video Generator</h1><p>Frontend not built.</p>")


@app.get("/health")
def health():
    return {"status": "ok"}
