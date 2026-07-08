"""ReminderAgent — local reminder and notification scheduling."""

from __future__ import annotations

import asyncio
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


class ReminderAgent(Agent):
    """Local reminder agent with persistence and auto-expiry.

    Supports create, read, update, delete, list, and dismiss operations.
    All data stored in local SQLite.
    """

    def __init__(self, db_path: str = "./memory_data/reminders.db") -> None:
        super().__init__(
            name="reminder",
            supported_task_types=(
                "reminder.create",
                "reminder.read",
                "reminder.update",
                "reminder.delete",
                "reminder.list",
                "reminder.dismiss",
            ),
        )
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS reminders (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                message TEXT NOT NULL DEFAULT '',
                due_at TEXT,
                priority INTEGER NOT NULL DEFAULT 0,
                dismissed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(due_at)")
        self._conn.commit()
        _logger.info("ReminderAgent initialised at '%s'", self._db_path)

    def _ensure_ready(self) -> None:
        if self._conn is None:
            raise RuntimeError("ReminderAgent not initialised. Call .initialize() first.")

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name, task_id=task.task_id,
                success=False, message=f"ReminderAgent cannot handle task type: {task.task_type}",
            )

        task_type = task.task_type
        try:
            if task_type == "reminder.create":
                return await self._create(task)
            if task_type == "reminder.read":
                return await self._read(task)
            if task_type == "reminder.update":
                return await self._update(task)
            if task_type == "reminder.delete":
                return await self._delete(task)
            if task_type == "reminder.list":
                return await self._list(task)
            if task_type == "reminder.dismiss":
                return await self._dismiss(task)
        except Exception as exc:
            _logger.exception("Reminder operation failed")
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=str(exc), data={"error": str(exc)})

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=f"Unknown reminder task: {task_type}")

    async def _create(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        reminder_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            "INSERT INTO reminders (id, title, message, due_at, priority, dismissed, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 0, ?, ?)",
            (reminder_id, str(task.payload.get("title", "")), str(task.payload.get("message", "")),
             task.payload.get("due_at"), int(task.payload.get("priority", 0)), now, now),
        )
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Reminder created", data={"id": reminder_id, "title": task.payload.get("title", "")})

    async def _read(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        reminder_id = task.payload.get("reminder_id", "")
        cursor = self._conn.execute("SELECT * FROM reminders WHERE id = ?", (reminder_id,))
        row = cursor.fetchone()

        if not row:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="Reminder not found")

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Reminder retrieved", data={
            "id": row[0], "title": row[1], "message": row[2], "due_at": row[3],
            "priority": row[4], "dismissed": bool(row[5]), "created_at": row[6], "updated_at": row[7],
        })

    async def _update(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        reminder_id = task.payload.get("reminder_id", "")

        updates: list[str] = []
        params: list[Any] = []
        for field in ("title", "message", "due_at", "priority"):
            if field in task.payload:
                updates.append(f"{field} = ?")
                params.append(task.payload[field])

        if not updates:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="No fields to update")

        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(reminder_id)

        cursor = self._conn.execute(f"UPDATE reminders SET {', '.join(updates)} WHERE id = ?", params)
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=cursor.rowcount > 0, message="Reminder updated" if cursor.rowcount > 0 else "Reminder not found")

    async def _delete(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        reminder_id = task.payload.get("reminder_id", "")
        cursor = self._conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=cursor.rowcount > 0, message="Reminder deleted" if cursor.rowcount > 0 else "Reminder not found")

    async def _list(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        include_dismissed = bool(task.payload.get("include_dismissed", False))
        limit = int(task.payload.get("limit", 50))

        if include_dismissed:
            cursor = self._conn.execute("SELECT * FROM reminders ORDER BY due_at ASC LIMIT ?", (limit,))
        else:
            cursor = self._conn.execute("SELECT * FROM reminders WHERE dismissed = 0 ORDER BY due_at ASC LIMIT ?", (limit,))

        rows = cursor.fetchall()
        reminders = [{"id": r[0], "title": r[1], "message": r[2], "due_at": r[3], "priority": r[4], "dismissed": bool(r[5])} for r in rows]

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Found {len(reminders)} reminders", data={"reminders": reminders, "count": len(reminders)})

    async def _dismiss(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        reminder_id = task.payload.get("reminder_id", "")
        now = datetime.now(timezone.utc).isoformat()

        cursor = self._conn.execute("UPDATE reminders SET dismissed = 1, updated_at = ? WHERE id = ? AND dismissed = 0", (now, reminder_id))
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=cursor.rowcount > 0, message="Reminder dismissed" if cursor.rowcount > 0 else "Reminder not found or already dismissed")
