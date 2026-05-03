"""Phase 5 endpoints — natural-language edit + intent classification."""
from __future__ import annotations
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agents.edit_agent import EditAgent
from shared.schemas.edit import EditCommand
from shared.utils.logging import get_logger
from state_manager.state_manager import StateManager

router = APIRouter()
log = get_logger("api.edit")
agent = EditAgent()
sm = StateManager()


class EditRequest(BaseModel):
    project_id: str
    query: str = Field(..., min_length=1)
    user_id: Optional[str] = None


class ClassifyRequest(BaseModel):
    query: str
    project_id: Optional[str] = None


@router.post("/classify")
def classify(req: ClassifyRequest):
    state = sm.latest(req.project_id) if req.project_id else None
    intent = agent.classify(EditCommand(project_id=req.project_id or "", query=req.query),
                            state=state)
    return intent.model_dump(mode="json")


@router.post("/apply")
def apply_edit(req: EditRequest):
    cmd = EditCommand(project_id=req.project_id, query=req.query, user_id=req.user_id)
    result = agent.edit(cmd)
    if not result.success:
        raise HTTPException(400, result.error or "edit failed")
    return result.model_dump(mode="json")


@router.get("/log/{project_id}")
def edit_log(project_id: str):
    return sm.edit_history(project_id)
