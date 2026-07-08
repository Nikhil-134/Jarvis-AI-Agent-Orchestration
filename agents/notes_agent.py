"""NotesAgent — persistent note-taking and retrieval.

Uses SQLite for local-first storage with no cloud dependencies.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask

_logger = logging.getLogger(__name__)


class NotesAgent(Agent):
    """Persistent note-taking agent.

    Supports create, read, update, delete, search, and list operations
    on notes stored in a local SQLite database.
    """

    def __init__(self, db_path: str = "./memory_data/notes.db") -> None:
        super().__init__(
            name="notes",
            supported_task_types=(
                "notes.create",
                "notes.read",
                "notes.update",
                "notes.delete",
                "notes.search",
                "notes.list",
                "notes.stats",
            ),
        )
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_title ON notes(title)")
        self._conn.commit()
        _logger.info("NotesAgent initialised at '%s'", self._db_path)

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name, task_id=task.task_id,
                success=False, message=f"NotesAgent cannot handle task type: {task.task_type}",
            )

        task_type = task.task_type

        try:
            if task_type == "notes.create":
                return await self._create(task)
            if task_type == "notes.read":
                return await self._read(task)
            if task_type == "notes.update":
                return await self._update(task)
            if task_type == "notes.delete":
                return await self._delete(task)
            if task_type == "notes.search":
                return await self._search(task)
            if task_type == "notes.list":
                return await self._list(task)
            if task_type == "notes.stats":
                return await self._stats(task)
        except Exception as exc:
            _logger.exception("Notes operation failed")
            return AgentResult(
                agent_name=self.name, task_id=task.task_id,
                success=False, message=str(exc), data={"error": str(exc)},
            )

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=f"Unknown notes task: {task_type}")

    def _ensure_ready(self) -> None:
        if self._conn is None:
            raise RuntimeError("NotesAgent not initialised. Call .initialize() first.")

    async def _create(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        note_id = str(uuid4())
        title = str(task.payload.get("title", ""))
        content = str(task.payload.get("content", ""))
        tags = json.dumps(task.payload.get("tags", []))
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            "INSERT INTO notes (id, title, content, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (note_id, title, content, tags, now, now),
        )
        self._conn.commit()

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message="Note created",
            data={"id": note_id, "title": title},
        )

    async def _read(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        note_id = task.payload.get("note_id", "")

        cursor = self._conn.execute("SELECT id, title, content, tags, created_at, updated_at FROM notes WHERE id = ?", (note_id,))
        row = cursor.fetchone()

        if not row:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="Note not found", data={"status": "not_found"})

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message="Note retrieved",
            data={"id": row[0], "title": row[1], "content": row[2], "tags": json.loads(row[3]), "created_at": row[4], "updated_at": row[5]},
        )

    async def _update(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        note_id = task.payload.get("note_id", "")

        updates: list[str] = []
        params: list[Any] = []
        for field in ("title", "content"):
            if field in task.payload:
                updates.append(f"{field} = ?")
                params.append(task.payload[field])

        if "tags" in task.payload:
            updates.append("tags = ?")
            params.append(json.dumps(task.payload["tags"]))

        if not updates:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="No fields to update")

        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(note_id)

        cursor = self._conn.execute(f"UPDATE notes SET {', '.join(updates)} WHERE id = ?", params)
        self._conn.commit()

        if cursor.rowcount == 0:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="Note not found")

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Note updated")

    async def _delete(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        note_id = task.payload.get("note_id", "")

        cursor = self._conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))
        self._conn.commit()

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=cursor.rowcount > 0,
            message="Note deleted" if cursor.rowcount > 0 else "Note not found",
            data={"deleted": cursor.rowcount > 0},
        )

    async def _search(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        query = str(task.payload.get("query", ""))
        limit = int(task.payload.get("limit", 20))

        cursor = self._conn.execute(
            "SELECT id, title, content, tags, created_at, updated_at FROM notes WHERE title LIKE ? OR content LIKE ? ORDER BY updated_at DESC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        )
        rows = cursor.fetchall()

        notes = [
            {"id": r[0], "title": r[1], "content": r[2][:200], "tags": json.loads(r[3]), "created_at": r[4], "updated_at": r[5]}
            for r in rows
        ]

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message=f"Found {len(notes)} notes",
            data={"notes": notes, "count": len(notes), "query": query},
        )

    async def _list(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        limit = int(task.payload.get("limit", 50))
        offset = int(task.payload.get("offset", 0))

        cursor = self._conn.execute("SELECT id, title, tags, created_at, updated_at FROM notes ORDER BY updated_at DESC LIMIT ? OFFSET ?", (limit, offset))
        rows = cursor.fetchall()

        notes = [
            {"id": r[0], "title": r[1], "tags": json.loads(r[2]), "created_at": r[3], "updated_at": r[4]}
            for r in rows
        ]

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message=f"Listed {len(notes)} notes",
            data={"notes": notes, "count": len(notes)},
        )

    async def _stats(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        cursor = self._conn.execute("SELECT COUNT(*) FROM notes")
        count = cursor.fetchone()[0]

        return AgentResult(
            agent_name=self.name, task_id=task.task_id,
            success=True, message="Notes stats",
            data={"total_notes": count},
        )
