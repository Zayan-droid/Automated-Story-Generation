"""Per-project WebSocket progress stream."""
from __future__ import annotations
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services import run_registry

router = APIRouter()


@router.websocket("/progress/{project_id}")
async def progress_ws(ws: WebSocket, project_id: str):
    await ws.accept()
    queue = run_registry.subscribe(project_id)
    try:
        # Initial snapshot so newly-connected clients catch up.
        snap = run_registry.snapshot(project_id)
        if snap:
            await ws.send_text(json.dumps({"type": "snapshot", "data": snap}))
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=30.0)
                await ws.send_text(json.dumps({"type": "event", "data": ev}))
                if ev.get("phase") in ("complete", "error"):
                    # Give the client time to read the final event before closing.
                    await asyncio.sleep(0.2)
                    break
            except asyncio.TimeoutError:
                # Heartbeat to keep the connection alive.
                await ws.send_text(json.dumps({"type": "heartbeat"}))
    except WebSocketDisconnect:
        pass
    finally:
        run_registry.unsubscribe(project_id, queue)
        try:
            await ws.close()
        except RuntimeError:
            pass
