"""SQLite-backed document store implementation."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from memory.exceptions import MemoryRetrievalError, MemoryStorageError
from memory.interfaces import IDocumentStore

_logger = logging.getLogger(__name__)


class SQLiteDocumentStore(IDocumentStore):
    """Persistent document store backed by SQLite.

    All operations are dispatched to a thread executor to avoid
    blocking the async event loop.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._conn: sqlite3.Connection | None = None

    async def initialize(self) -> None:
        """Open the database and create tables if they do not exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        def _init() -> sqlite3.Connection:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collection ON documents(collection)")
            conn.commit()
            return conn

        loop = asyncio.get_running_loop()
        self._conn = await loop.run_in_executor(None, _init)
        _logger.info("SQLiteDocumentStore initialised at '%s'", self._path)

    async def store(self, collection: str, document: dict[str, Any]) -> str:
        self._ensure_ready()

        doc_id = document.get("id", str(__import__("uuid").uuid4()))
        try:
            loop = asyncio.get_running_loop()

            def _store() -> None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO documents (id, collection, content, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        doc_id,
                        collection,
                        json.dumps(document, default=str),
                        document.get("created_at", ""),
                    ),
                )
                self._conn.commit()

            await loop.run_in_executor(None, _store)
            _logger.debug("Stored document '%s' in collection '%s'", doc_id, collection)
            return doc_id
        except Exception as exc:
            raise MemoryStorageError(f"Failed to store document: {exc}") from exc

    async def retrieve(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        self._ensure_ready()

        try:
            loop = asyncio.get_running_loop()

            def _retrieve() -> dict[str, Any] | None:
                cursor = self._conn.execute(
                    "SELECT content FROM documents WHERE id = ? AND collection = ?",
                    (doc_id, collection),
                )
                row = cursor.fetchone()
                if row is None:
                    return None
                return json.loads(row[0])

            return await loop.run_in_executor(None, _retrieve)
        except Exception as exc:
            raise MemoryRetrievalError(f"Failed to retrieve document: {exc}") from exc

    async def search(
        self,
        collection: str,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        self._ensure_ready()

        try:
            loop = asyncio.get_running_loop()

            def _search() -> list[dict[str, Any]]:
                if filters:
                    where_clauses = [f"json_extract(content, '$.{k}') = ?" for k in filters]
                    where_sql = " AND ".join(where_clauses)
                    params = [json.dumps(v) if not isinstance(v, str) else v for v in filters.values()]
                    cursor = self._conn.execute(
                        f"SELECT content FROM documents WHERE collection = ? AND {where_sql} ORDER BY rowid DESC LIMIT ?",
                        [collection, *params, limit],
                    )
                else:
                    cursor = self._conn.execute(
                        "SELECT content FROM documents WHERE collection = ? ORDER BY rowid DESC LIMIT ?",
                        (collection, limit),
                    )
                return [json.loads(row[0]) for row in cursor.fetchall()]

            return await loop.run_in_executor(None, _search)
        except Exception as exc:
            raise MemoryRetrievalError(f"Failed to search documents: {exc}") from exc

    async def delete(self, collection: str, doc_id: str) -> bool:
        self._ensure_ready()

        try:
            loop = asyncio.get_running_loop()

            def _delete() -> bool:
                cursor = self._conn.execute(
                    "DELETE FROM documents WHERE id = ? AND collection = ?",
                    (doc_id, collection),
                )
                self._conn.commit()
                return cursor.rowcount > 0

            result = await loop.run_in_executor(None, _delete)
            if result:
                _logger.debug("Deleted document '%s' from collection '%s'", doc_id, collection)
            return result
        except Exception as exc:
            raise MemoryStorageError(f"Failed to delete document: {exc}") from exc

    async def count(self, collection: str | None = None) -> int:
        """Return the number of documents, optionally filtered by collection."""
        self._ensure_ready()

        loop = asyncio.get_running_loop()

        def _count() -> int:
            if collection:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE collection = ?", (collection,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM documents")
            return cursor.fetchone()[0]

        return await loop.run_in_executor(None, _count)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:

            def _close() -> None:
                self._conn.close()

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, _close)
            self._conn = None
            _logger.info("SQLiteDocumentStore closed")

    def _ensure_ready(self) -> None:
        if self._conn is None:
            raise RuntimeError(
                "SQLiteDocumentStore not initialised. Call .initialize() first."
            )
