"""SQLite-backed append-only version log."""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Dict, Any

from shared import constants


SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    TEXT NOT NULL,
    version       INTEGER NOT NULL,
    parent_version INTEGER,
    created_at    TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    state_path    TEXT NOT NULL,
    asset_paths   TEXT NOT NULL DEFAULT '[]',
    edit_intent   TEXT,
    UNIQUE(project_id, version)
);

CREATE INDEX IF NOT EXISTS idx_versions_project ON versions(project_id);

CREATE TABLE IF NOT EXISTS edit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    query       TEXT NOT NULL,
    intent_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_edit_log_project ON edit_log(project_id);
"""


class SqliteStorage:
    """Thin wrapper around the version log database."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else Path(constants.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def append_version(
        self,
        project_id: str,
        version: int,
        state_path: str,
        asset_paths: List[str],
        description: str = "",
        parent_version: Optional[int] = None,
        edit_intent: Optional[Dict[str, Any]] = None,
        created_at: str = "",
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO versions
                   (project_id, version, parent_version, created_at,
                    description, state_path, asset_paths, edit_intent)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    project_id,
                    version,
                    parent_version,
                    created_at,
                    description,
                    state_path,
                    json.dumps(asset_paths),
                    json.dumps(edit_intent) if edit_intent else None,
                ),
            )
            return cur.lastrowid

    def get_version(self, project_id: str, version: int) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM versions WHERE project_id=? AND version=?",
                (project_id, version),
            ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_versions(self, project_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM versions WHERE project_id=? ORDER BY version ASC",
                (project_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def latest_version(self, project_id: str) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT MAX(version) AS v FROM versions WHERE project_id=?",
                (project_id,),
            ).fetchone()
        return int(row["v"] or 0)

    def list_projects(self) -> List[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT DISTINCT project_id FROM versions ORDER BY project_id DESC"
            ).fetchall()
        return [r["project_id"] for r in rows]

    def log_edit(
        self,
        project_id: str,
        query: str,
        intent: Dict[str, Any],
        result: Dict[str, Any],
        created_at: str,
    ) -> int:
        with self._conn() as c:
            cur = c.execute(
                """INSERT INTO edit_log (project_id, created_at, query, intent_json, result_json)
                   VALUES (?,?,?,?,?)""",
                (project_id, created_at, query, json.dumps(intent), json.dumps(result)),
            )
            return cur.lastrowid

    def list_edits(self, project_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM edit_log WHERE project_id=? ORDER BY id DESC",
                (project_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "project_id": r["project_id"],
                "created_at": r["created_at"],
                "query": r["query"],
                "intent": json.loads(r["intent_json"]),
                "result": json.loads(r["result_json"]),
            }
            for r in rows
        ]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "version": row["version"],
            "parent_version": row["parent_version"],
            "created_at": row["created_at"],
            "description": row["description"],
            "state_path": row["state_path"],
            "asset_paths": json.loads(row["asset_paths"]),
            "edit_intent": json.loads(row["edit_intent"]) if row["edit_intent"] else None,
        }
