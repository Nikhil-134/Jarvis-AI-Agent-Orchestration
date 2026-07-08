"""CalendarAgent — local-first calendar and event management."""

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


class CalendarAgent(Agent):
    """Local calendar agent for event management.

    Supports create, read, update, delete, list, and search operations.
    All data stored in local SQLite — no cloud dependency.
    """

    def __init__(self, db_path: str = "./memory_data/calendar.db") -> None:
        super().__init__(
            name="calendar",
            supported_task_types=(
                "calendar.create",
                "calendar.read",
                "calendar.update",
                "calendar.delete",
                "calendar.list",
                "calendar.search",
            ),
        )
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                all_day INTEGER NOT NULL DEFAULT 0,
                location TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_start ON events(start_time)")
        self._conn.commit()
        _logger.info("CalendarAgent initialised at '%s'", self._db_path)

    def _ensure_ready(self) -> None:
        if self._conn is None:
            raise RuntimeError("CalendarAgent not initialised. Call .initialize() first.")

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name, task_id=task.task_id,
                success=False, message=f"CalendarAgent cannot handle task type: {task.task_type}",
            )

        task_type = task.task_type
        try:
            if task_type == "calendar.create":
                return await self._create(task)
            if task_type == "calendar.read":
                return await self._read(task)
            if task_type == "calendar.update":
                return await self._update(task)
            if task_type == "calendar.delete":
                return await self._delete(task)
            if task_type == "calendar.list":
                return await self._list(task)
            if task_type == "calendar.search":
                return await self._search(task)
        except Exception as exc:
            _logger.exception("Calendar operation failed")
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=str(exc), data={"error": str(exc)})

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=f"Unknown calendar task: {task_type}")

    async def _create(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        event_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            "INSERT INTO events (id, title, description, start_time, end_time, all_day, location, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, str(task.payload.get("title", "")), str(task.payload.get("description", "")),
             str(task.payload.get("start_time", now)), str(task.payload.get("end_time", now)),
             int(task.payload.get("all_day", 0)), str(task.payload.get("location", "")), now, now),
        )
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Event created", data={"id": event_id, "title": task.payload.get("title", "")})

    async def _read(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        event_id = task.payload.get("event_id", "")
        cursor = self._conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cursor.fetchone()

        if not row:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="Event not found")

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Event retrieved", data={
            "id": row[0], "title": row[1], "description": row[2],
            "start_time": row[3], "end_time": row[4], "all_day": bool(row[5]),
            "location": row[6], "created_at": row[7], "updated_at": row[8],
        })

    async def _update(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        event_id = task.payload.get("event_id", "")

        updates: list[str] = []
        params: list[Any] = []
        for field in ("title", "description", "start_time", "end_time", "location"):
            if field in task.payload:
                updates.append(f"{field} = ?")
                params.append(task.payload[field])
        if "all_day" in task.payload:
            updates.append("all_day = ?")
            params.append(int(task.payload["all_day"]))

        if not updates:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="No fields to update")

        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(event_id)

        cursor = self._conn.execute(f"UPDATE events SET {', '.join(updates)} WHERE id = ?", params)
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=cursor.rowcount > 0, message="Event updated" if cursor.rowcount > 0 else "Event not found")

    async def _delete(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        event_id = task.payload.get("event_id", "")
        cursor = self._conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=cursor.rowcount > 0, message="Event deleted" if cursor.rowcount > 0 else "Event not found")

    async def _list(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        start_date = str(task.payload.get("start_date", ""))
        end_date = str(task.payload.get("end_date", ""))
        limit = int(task.payload.get("limit", 50))

        if start_date and end_date:
            cursor = self._conn.execute(
                "SELECT * FROM events WHERE start_time >= ? AND start_time <= ? ORDER BY start_time ASC LIMIT ?",
                (start_date, end_date, limit),
            )
        else:
            cursor = self._conn.execute("SELECT * FROM events ORDER BY start_time ASC LIMIT ?", (limit,))

        rows = cursor.fetchall()
        events = [{"id": r[0], "title": r[1], "description": r[2], "start_time": r[3], "end_time": r[4], "all_day": bool(r[5]), "location": r[6]} for r in rows]

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Found {len(events)} events", data={"events": events, "count": len(events)})

    async def _search(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        query = str(task.payload.get("query", ""))
        limit = int(task.payload.get("limit", 20))

        cursor = self._conn.execute(
            "SELECT * FROM events WHERE title LIKE ? OR description LIKE ? ORDER BY start_time ASC LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        )
        rows = cursor.fetchall()
        events = [{"id": r[0], "title": r[1], "description": r[2], "start_time": r[3], "end_time": r[4], "all_day": bool(r[5]), "location": r[6]} for r in rows]

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Found {len(events)} events", data={"events": events, "count": len(events)})
