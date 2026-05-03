"""In-memory registry of in-flight runs and their latest event.

Lets the WebSocket layer subscribe + the REST status endpoint poll without a
real broker. Suitable for single-process deploys; for prod swap to Redis.
"""
from __future__ import annotations
import asyncio
import threading
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional


_state_lock = threading.Lock()
_runs: Dict[str, Dict] = {}                 # project_id -> dict
_events: Dict[str, Deque[Dict]] = defaultdict(lambda: deque(maxlen=200))
_subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)


def create(project_id: str) -> None:
    with _state_lock:
        _runs[project_id] = {"project_id": project_id, "status": "running",
                             "phase": "queued", "progress": 0.0,
                             "message": "queued", "events": 0}
        _events[project_id].clear()


def push_event(project_id: str, event: Dict) -> None:
    with _state_lock:
        _events[project_id].append(event)
        run = _runs.setdefault(project_id, {"project_id": project_id, "events": 0})
        run.update({
            "phase": event.get("phase", run.get("phase")),
            "status": event.get("status", run.get("status")),
            "progress": event.get("progress", run.get("progress", 0.0)),
            "message": event.get("message", ""),
            "events": run.get("events", 0) + 1,
        })
    # Fan out to all WS subscribers.
    for q in list(_subscribers.get(project_id, [])):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass


def snapshot(project_id: str) -> Optional[Dict]:
    with _state_lock:
        run = _runs.get(project_id)
        if not run:
            return None
        return {**run, "recent_events": list(_events[project_id])[-20:]}


def history(project_id: str) -> List[Dict]:
    with _state_lock:
        return list(_events.get(project_id, []))


def subscribe(project_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers[project_id].append(q)
    # Replay recent events.
    for ev in list(_events[project_id]):
        try:
            q.put_nowait(ev)
        except asyncio.QueueFull:
            break
    return q


def unsubscribe(project_id: str, q: asyncio.Queue) -> None:
    if q in _subscribers.get(project_id, []):
        _subscribers[project_id].remove(q)
