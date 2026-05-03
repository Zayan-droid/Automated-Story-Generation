"""History formatting helpers — turns raw version rows into UI-friendly diffs."""
from __future__ import annotations
from typing import List, Dict, Any


def diff_summary(prev: Dict[str, Any] | None, curr: Dict[str, Any]) -> str:
    """One-line description of what changed in this version."""
    if prev is None:
        return curr.get("description") or "initial pipeline run"
    desc = curr.get("description") or ""
    if desc:
        return desc
    intent = curr.get("edit_intent") or {}
    if intent:
        return f"{intent.get('intent','edit')} → {intent.get('target','')}"
    return "snapshot"


def format_history(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    prev = None
    for row in rows:
        out.append(
            {
                "version": row["version"],
                "created_at": row["created_at"],
                "description": diff_summary(prev, row),
                "edit_intent": row.get("edit_intent"),
                "parent_version": row.get("parent_version"),
                "asset_count": len(row.get("asset_paths") or []),
            }
        )
        prev = row
    return out
