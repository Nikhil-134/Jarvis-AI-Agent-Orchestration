"""EmailAgent — local email generation and management.

Generates and drafts emails locally.  Sending requires an SMTP
configuration but drafting is fully local.  All drafts stored in
SQLite.
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


class EmailAgent(Agent):
    """Local email agent for drafting and managing emails.

    Supports create draft, read, update, delete, list, and send operations.
    Drafts are stored locally in SQLite.  Sending connects to an SMTP
    server configured in settings.
    """

    def __init__(self, db_path: str = "./memory_data/emails.db") -> None:
        super().__init__(
            name="email",
            supported_task_types=(
                "email.draft",
                "email.read",
                "email.update",
                "email.delete",
                "email.list",
                "email.send",
            ),
        )
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS emails (
                id TEXT PRIMARY KEY,
                to_addr TEXT NOT NULL,
                cc TEXT NOT NULL DEFAULT '',
                bcc TEXT NOT NULL DEFAULT '',
                subject TEXT NOT NULL,
                body TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'draft',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self._conn.commit()
        _logger.info("EmailAgent initialised at '%s'", self._db_path)

    def _ensure_ready(self) -> None:
        if self._conn is None:
            raise RuntimeError("EmailAgent not initialised. Call .initialize() first.")

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name, task_id=task.task_id,
                success=False, message=f"EmailAgent cannot handle task type: {task.task_type}",
            )

        task_type = task.task_type
        try:
            if task_type == "email.draft":
                return await self._draft(task)
            if task_type == "email.read":
                return await self._read(task)
            if task_type == "email.update":
                return await self._update(task)
            if task_type == "email.delete":
                return await self._delete(task)
            if task_type == "email.list":
                return await self._list(task)
            if task_type == "email.send":
                return await self._send(task)
        except Exception as exc:
            _logger.exception("Email operation failed")
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=str(exc), data={"error": str(exc)})

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=f"Unknown email task: {task_type}")

    async def _draft(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        email_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            "INSERT INTO emails (id, to_addr, cc, bcc, subject, body, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, ?)",
            (email_id, str(task.payload.get("to", "")), str(task.payload.get("cc", "")),
             str(task.payload.get("bcc", "")), str(task.payload.get("subject", "")),
             str(task.payload.get("body", "")), now, now),
        )
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Email draft created", data={"id": email_id, "subject": task.payload.get("subject", "")})

    async def _read(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        email_id = task.payload.get("email_id", "")
        cursor = self._conn.execute("SELECT * FROM emails WHERE id = ?", (email_id,))
        row = cursor.fetchone()

        if not row:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="Email not found")

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Email retrieved", data={
            "id": row[0], "to": row[1], "cc": row[2], "bcc": row[3],
            "subject": row[4], "body": row[5], "status": row[6], "created_at": row[7], "updated_at": row[8],
        })

    async def _update(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        email_id = task.payload.get("email_id", "")

        updates: list[str] = []
        params: list[Any] = []
        for field in ("to_addr", "cc", "bcc", "subject", "body"):
            payload_key = "to" if field == "to_addr" else field
            if payload_key in task.payload:
                updates.append(f"{field} = ?")
                params.append(task.payload[payload_key])

        if not updates:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="No fields to update")

        updates.append("updated_at = ?")
        params.append(datetime.now(timezone.utc).isoformat())
        params.append(email_id)

        cursor = self._conn.execute(f"UPDATE emails SET {', '.join(updates)} WHERE id = ?", params)
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=cursor.rowcount > 0, message="Email updated" if cursor.rowcount > 0 else "Email not found")

    async def _delete(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        email_id = task.payload.get("email_id", "")
        cursor = self._conn.execute("DELETE FROM emails WHERE id = ?", (email_id,))
        self._conn.commit()

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=cursor.rowcount > 0, message="Email deleted" if cursor.rowcount > 0 else "Email not found")

    async def _list(self, task: AgentTask) -> AgentResult:
        self._ensure_ready()
        status = task.payload.get("status", "")
        limit = int(task.payload.get("limit", 50))

        if status:
            cursor = self._conn.execute("SELECT id, to_addr, subject, status, created_at FROM emails WHERE status = ? ORDER BY created_at DESC LIMIT ?", (status, limit))
        else:
            cursor = self._conn.execute("SELECT id, to_addr, subject, status, created_at FROM emails ORDER BY created_at DESC LIMIT ?", (limit,))

        rows = cursor.fetchall()
        emails = [{"id": r[0], "to": r[1], "subject": r[2], "status": r[3], "created_at": r[4]} for r in rows]

        return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message=f"Found {len(emails)} emails", data={"emails": emails, "count": len(emails)})

    async def _send(self, task: AgentTask) -> AgentResult:
        email_id = task.payload.get("email_id", "")
        if not email_id:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="No email_id provided")

        self._ensure_ready()
        cursor = self._conn.execute("SELECT * FROM emails WHERE id = ?", (email_id,))
        row = cursor.fetchone()

        if not row:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="Email not found")

        try:
            import smtplib
            from email.message import EmailMessage

            config = task.payload.get("smtp", {})
            smtp_host = config.get("host", task.payload.get("smtp_host", "localhost"))
            smtp_port = int(config.get("port", task.payload.get("smtp_port", 25)))
            smtp_user = config.get("user", task.payload.get("smtp_user", ""))
            smtp_pass = config.get("password", task.payload.get("smtp_password", ""))

            msg = EmailMessage()
            msg["Subject"] = row[4]
            msg["From"] = config.get("from_addr", smtp_user)
            msg["To"] = row[1]
            if row[2]:
                msg["Cc"] = row[2]
            msg.set_content(row[5])

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if smtp_user:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                server.send_message(msg)

            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute("UPDATE emails SET status = 'sent', updated_at = ? WHERE id = ?", (now, email_id))
            self._conn.commit()

            return AgentResult(agent_name=self.name, task_id=task.task_id, success=True, message="Email sent", data={"id": email_id, "to": row[1], "subject": row[4]})

        except ImportError:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message="smtplib not available")
        except Exception as exc:
            return AgentResult(agent_name=self.name, task_id=task.task_id, success=False, message=f"Failed to send email: {exc}", data={"error": str(exc)})
