"""Background pipeline runner — wraps the orchestrator + pushes events to the registry."""
from __future__ import annotations
from typing import Optional

from agents.orchestrator import PipelineOrchestrator, ProgressEvent
from shared.utils.logging import get_logger

from . import run_registry

log = get_logger("pipeline_service")
_orchestrator = PipelineOrchestrator()


def run_full_async(prompt: str, project_id: str, target_duration_s: int = 45,
                   scene_count: int = 4, with_bgm: bool = True,
                   with_subtitles: bool = True, subtitle_language: str = "English") -> None:
    def push(ev: ProgressEvent) -> None:
        run_registry.push_event(project_id, {
            "phase": ev.phase, "status": ev.status, "message": ev.message,
            "progress": ev.progress, "project_id": project_id,
            "payload": ev.payload,
        })

    try:
        _orchestrator.run_full(
            prompt=prompt,
            project_id=project_id,
            target_duration_s=target_duration_s,
            scene_count=scene_count,
            with_bgm=with_bgm,
            with_subtitles=with_subtitles,
            subtitle_language=subtitle_language,
            on_event=push,
        )
    except Exception as e:  # noqa: BLE001
        log.exception("pipeline run failed")
        run_registry.push_event(project_id, {
            "phase": "error", "status": "failed",
            "message": f"{type(e).__name__}: {e}", "progress": 1.0,
            "project_id": project_id,
        })


def rerun_phase_async(project_id: str, phase: str) -> None:
    def push(ev: ProgressEvent) -> None:
        run_registry.push_event(project_id, {
            "phase": ev.phase, "status": ev.status, "message": ev.message,
            "progress": ev.progress, "project_id": project_id,
            "payload": ev.payload,
        })
    try:
        _orchestrator.re_run_phase(project_id, phase, on_event=push)
    except Exception as e:  # noqa: BLE001
        log.exception("phase rerun failed")
        run_registry.push_event(project_id, {
            "phase": "error", "status": "failed",
            "message": f"{type(e).__name__}: {e}", "progress": 1.0,
            "project_id": project_id,
        })
