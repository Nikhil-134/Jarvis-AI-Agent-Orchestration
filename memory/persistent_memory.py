"""Persistent memory — cross-session recall for JARVIS.

This is the layer that makes JARVIS *remember*: conversations, long-running
projects, and user preferences survive a terminal close or a Windows restart,
and reflection turns finished conversations into durable insight.

It is a thin, single-responsibility orchestration layer built entirely on top of
the existing :class:`~memory.memory_manager.MemoryManager` (dependency-injected).
It adds no new storage backend — durability comes for free from the manager's
SQLite document store and ChromaDB vector store. What it adds is *structure*:

* **Sessions** — every turn is tagged with a ``session_id`` so a session can be
  restored chronologically ("continue where I left off").
* **Projects** — named, status-bearing memories with upsert semantics.
* **User profile** — key/value preferences with upsert + fast dictionary recall.
* **Reflection** — decisions / tasks / lessons / summary distilled per session.
* **Classification + a store-worthiness gate** — meaningful information only.

Layer map (responsibilities)::

    Working memory     MemoryManager.working_memory (RAM, restored on demand)
    Conversation       record_turn / restore_session / recent_turns
    Project            remember_project / get_project / list_projects
    User profile       set_preference / get_preference / get_profile
    Long-term + vector MemoryManager (durable SQLite + ChromaDB, dedup, ranking)
    Reflection         reflect_on_session (+ ReflectionEngine)
    Knowledge graph    (future — see PROJECT_BRAIN roadmap)

All identifiers are sanitised (:mod:`memory.validation`) and every stored object
is validated, so poisoned or oversized input cannot corrupt the store.
"""

from __future__ import annotations

import logging
from typing import Any

from memory.memory_manager import MemoryManager
from memory.memory_service import MemoryService
from memory.models import MemoryItem, MemoryType, calculate_importance
from memory.preference_extractor import PreferenceExtractor
from memory.reflection import Reflection, ReflectionEngine
from memory.validation import (
    sanitize_identifier,
    sanitize_text,
    to_safe_context_block,
    validate_memory_item,
)

_logger = logging.getLogger(__name__)

# Deterministic id prefixes give upsert semantics (a re-save replaces the row).
_PROJECT_ID_PREFIX = "project:"
_PREF_ID_PREFIX = "pref:"

# Classification cue words.
_PREFERENCE_CUES = ("i prefer", "i like", "my favourite", "my favorite", "please always", "please use", "i want you to")
_DECISION_CUES = ("we decided", "let's go with", "we'll use", "we will use", "decision:", "agreed to")
_TASK_CUES = ("todo", "to do", "need to", "next step", "remind me to", "i have to", "task:")
_IDEA_CUES = ("idea:", "what if", "maybe we could", "brainstorm", "could try")


class PersistentMemoryService:
    """Cross-session memory orchestration over a :class:`MemoryManager`.

    The manager MUST already be initialised (``await manager.initialize()``)
    before this service is used. The service itself is stateless beyond its
    configuration, so it is safe to share across the app.
    """

    def __init__(
        self,
        manager: MemoryManager,
        reflection_engine: ReflectionEngine | None = None,
        *,
        session_restore_limit: int = 20,
        recent_scan_limit: int = 500,
        preference_extractor: PreferenceExtractor | None = None,
    ) -> None:
        self._manager = manager
        self._reflection = reflection_engine or ReflectionEngine()
        self._session_restore_limit = session_restore_limit
        self._recent_scan_limit = recent_scan_limit
        # Deterministic, local preference extraction (roadmap #11). Injectable
        # for testing; the default is a stateless regex extractor.
        self._pref_extractor = preference_extractor or PreferenceExtractor()

    @property
    def manager(self) -> MemoryManager:
        """Expose the underlying manager for advanced/semantic operations."""
        return self._manager

    # ==================================================================
    # Conversation / session memory
    # ==================================================================

    async def record_turn(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        *,
        importance: float | None = None,
    ) -> str:
        """Persist one conversation turn under *session_id*.

        Returns the stored memory id, or ``""`` if the turn is not worth
        remembering (empty/boilerplate/tool-JSON — reusing the proven
        store-worthiness gate).
        """
        sid = sanitize_identifier(session_id, field="session_id")
        user_text = sanitize_text(user_text)
        assistant_text = sanitize_text(assistant_text)

        if not MemoryService._is_storeworthy(user_text, assistant_text):
            _logger.debug("Skipped low-value turn for session '%s'", sid)
            return ""

        content = f"User: {user_text}\nJARVIS: {assistant_text}"
        imp = importance if importance is not None else calculate_importance(content)
        item = MemoryItem(
            content=content,
            memory_type=MemoryType.CONVERSATION,
            importance=imp,
            metadata={
                "kind": "turn",
                "session_id": sid,
                "user": user_text[:500],
                "assistant_preview": assistant_text[:200],
            },
        )
        validate_memory_item(item)
        return await self._manager.store(item)

    async def restore_session(
        self, session_id: str, limit: int | None = None
    ) -> list[MemoryItem]:
        """Reload a session's recent turns from durable storage, oldest→newest.

        Also re-populates working memory, so the running conversation can pick
        up exactly where it left off after a restart.
        """
        sid = sanitize_identifier(session_id, field="session_id")
        cap = limit or self._session_restore_limit
        turns = await self._session_turns(sid, limit=cap)
        for item in turns:  # chronological → working memory keeps correct order
            self._manager.working_memory.add(item)
        _logger.info("Restored %d turn(s) for session '%s'", len(turns), sid)
        return turns

    async def recent_turns(self, limit: int = 10) -> list[MemoryItem]:
        """Return the most recent conversation turns across all sessions, newest first."""
        items = await self._manager.recent(MemoryType.CONVERSATION, limit=limit)
        return [it for it in items if it.metadata.get("kind") == "turn"]

    async def session_transcript(self, session_id: str, limit: int = 100) -> str:
        """Return a session's turns joined into a plain-text transcript."""
        turns = await self._session_turns(sanitize_identifier(session_id, field="session_id"), limit=limit)
        return "\n\n".join(t.content for t in turns)

    async def _session_turns(self, sid: str, limit: int) -> list[MemoryItem]:
        """Fetch a session's turns from the durable store, ordered oldest→newest."""
        scanned = await self._manager.recent(MemoryType.CONVERSATION, limit=self._recent_scan_limit)
        session_items = [
            it for it in scanned
            if it.metadata.get("kind") == "turn" and it.metadata.get("session_id") == sid
        ]
        # ``recent`` is newest-first; keep the newest ``limit`` then flip to
        # chronological order for natural replay.
        session_items = session_items[:limit]
        session_items.reverse()
        return session_items

    # ==================================================================
    # Project memory
    # ==================================================================

    async def remember_project(
        self,
        project_id: str,
        name: str,
        content: str,
        *,
        status: str = "active",
        importance: float = 0.9,
    ) -> str:
        """Create or update (upsert) a long-running project memory."""
        pid = sanitize_identifier(project_id, field="project_id")
        name = sanitize_text(name, max_chars=200)
        content = sanitize_text(content)
        status = sanitize_text(status, max_chars=40) or "active"

        doc_id = f"{_PROJECT_ID_PREFIX}{pid}"
        item = MemoryItem(
            id=doc_id,
            content=f"Project '{name}' [{status}]: {content}",
            memory_type=MemoryType.PROJECT,
            importance=importance,
            metadata={"kind": "project", "project_id": pid, "name": name, "status": status},
        )
        validate_memory_item(item)
        await self._upsert(item)
        _logger.info("Stored project '%s' (status=%s)", pid, status)
        return doc_id

    async def get_project(self, project_id: str) -> MemoryItem | None:
        """Return a project memory by id, or None."""
        pid = sanitize_identifier(project_id, field="project_id")
        projects = await self._manager.recent(MemoryType.PROJECT, limit=self._recent_scan_limit)
        for it in projects:
            if it.metadata.get("project_id") == pid:
                return it
        return None

    async def list_projects(
        self, *, status: str | None = None, limit: int = 50
    ) -> list[MemoryItem]:
        """List project memories, optionally filtered by *status*, newest first."""
        projects = await self._manager.recent(MemoryType.PROJECT, limit=self._recent_scan_limit)
        if status is not None:
            projects = [p for p in projects if p.metadata.get("status") == status]
        return projects[:limit]

    async def update_project_status(self, project_id: str, status: str) -> bool:
        """Update a project's status in place. Returns False if unknown."""
        existing = await self.get_project(project_id)
        if existing is None:
            return False
        meta = existing.metadata
        # Recover the original name/content from the stored composite content.
        content = existing.content.split(": ", 1)[1] if ": " in existing.content else existing.content
        await self.remember_project(
            project_id, meta.get("name", project_id), content,
            status=status, importance=existing.importance,
        )
        return True

    # ==================================================================
    # User profile / preferences
    # ==================================================================

    async def set_preference(self, key: str, value: str, *, importance: float = 0.85) -> str:
        """Store or update (upsert) a single user preference."""
        pref_key = sanitize_identifier(key, field="preference key")
        value = sanitize_text(value, max_chars=500)
        doc_id = f"{_PREF_ID_PREFIX}{pref_key}"
        item = MemoryItem(
            id=doc_id,
            content=f"User preference — {pref_key}: {value}",
            memory_type=MemoryType.PREFERENCE,
            importance=importance,
            metadata={"kind": "preference", "key": pref_key, "value": value},
        )
        validate_memory_item(item)
        await self._upsert(item)
        _logger.info("Stored preference '%s'", pref_key)
        return doc_id

    async def get_preference(self, key: str) -> str | None:
        """Return the current value of a preference, or None."""
        pref_key = sanitize_identifier(key, field="preference key")
        prefs = await self._manager.recent(MemoryType.PREFERENCE, limit=self._recent_scan_limit)
        for it in prefs:  # newest-first → first match is the latest value
            if it.metadata.get("key") == pref_key:
                return it.metadata.get("value")
        return None

    async def get_profile(self) -> dict[str, str]:
        """Return the full user profile as a ``{key: value}`` dict (latest wins)."""
        prefs = await self._manager.recent(MemoryType.PREFERENCE, limit=self._recent_scan_limit)
        profile: dict[str, str] = {}
        for it in prefs:  # newest-first
            key = it.metadata.get("key")
            if key and key not in profile:
                profile[key] = it.metadata.get("value", "")
        return profile

    async def learn_preferences(self, user_text: str) -> list[str]:
        """Auto-promote preferences stated in *user_text* to structured profile.

        Closes roadmap #11: identity/preference statements said in ordinary
        conversation ("call me Boss", "my favourite language is Rust", "I live
        in Bangalore") are promoted to durable, exact-recall ``set_preference``
        entries — no explicit "remember" command required. Deterministic and
        local (regex :class:`PreferenceExtractor`); the LLM is never involved.

        Returns the list of preference keys stored (may be empty). Best-effort:
        an individual store failure is logged and skipped — one bad preference
        must never break the others or the caller.
        """
        try:
            extracted = self._pref_extractor.extract(user_text or "")
        except Exception:  # noqa: BLE001 - extraction must never raise upward
            _logger.debug("Preference extraction failed", exc_info=True)
            return []

        stored: list[str] = []
        for pref in extracted:
            try:
                await self.set_preference(pref.key, pref.value)
                stored.append(pref.key)
            except Exception:  # noqa: BLE001 - skip a single unusable preference
                _logger.debug("Failed to store preference '%s'", pref.key, exc_info=True)
        if stored:
            _logger.info("Auto-learned %d preference(s): %s", len(stored), ", ".join(stored))
        return stored

    # ==================================================================
    # Reflection
    # ==================================================================

    async def reflect_on_session(self, session_id: str, *, limit: int = 100) -> list[MemoryItem]:
        """Distil a session into durable decisions/tasks/lessons/summary memories.

        Returns the list of newly stored reflection memories (may be empty).
        """
        sid = sanitize_identifier(session_id, field="session_id")
        transcript = await self.session_transcript(sid, limit=limit)
        if not transcript.strip():
            return []

        reflection: Reflection = await self._reflection.reflect(transcript)
        if reflection.is_empty():
            return []

        stored: list[MemoryItem] = []
        base_meta = {"session_id": sid, "reflection_source": reflection.source}

        if reflection.summary:
            stored.append(await self._store_reflection(
                reflection.summary, MemoryType.SUMMARY, 0.8, {**base_meta, "kind": "summary"}))
        for decision in reflection.decisions:
            stored.append(await self._store_reflection(
                decision, MemoryType.DECISION, 0.85, {**base_meta, "kind": "decision"}))
        for taskn in reflection.tasks:
            stored.append(await self._store_reflection(
                taskn, MemoryType.TASK, 0.75, {**base_meta, "kind": "task"}))
        for lesson in reflection.lessons:
            stored.append(await self._store_reflection(
                lesson, MemoryType.REFLECTION, 0.8, {**base_meta, "kind": "lesson"}))

        result = [it for it in stored if it is not None]
        _logger.info("Reflection stored %d insight(s) for session '%s' (source=%s)",
                     len(result), sid, reflection.source)
        return result

    async def _store_reflection(
        self, text: str, mtype: MemoryType, importance: float, metadata: dict[str, Any]
    ) -> MemoryItem | None:
        item = MemoryItem(content=text, memory_type=mtype, importance=importance, metadata=metadata)
        try:
            validate_memory_item(item)
        except Exception:  # noqa: BLE001 - skip an individual unusable insight
            return None
        await self._manager.store(item)
        return item

    # ==================================================================
    # Retrieval / classification
    # ==================================================================

    async def search(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        """Semantic search across all persistent memory."""
        return await self._manager.search(sanitize_text(query, max_chars=2000), top_k=top_k)

    async def build_context(self, query: str, top_k: int = 5) -> str:
        """Return relevant memories as an injection-safe, delimited context block."""
        items = await self.search(query, top_k=top_k)
        return to_safe_context_block([it.content for it in items])

    async def remember(
        self,
        text: str,
        *,
        memory_type: MemoryType | None = None,
        importance: float | None = None,
    ) -> str:
        """Auto-classify and store *text* if it is meaningful.

        Returns the stored id, or ``""`` when the text is judged not worth
        remembering (so JARVIS does not store everything).
        """
        clean = sanitize_text(text)
        if not self.is_meaningful(clean):
            return ""
        mtype = memory_type or self.classify(clean)
        imp = importance if importance is not None else calculate_importance(clean, mtype)
        item = MemoryItem(
            content=clean, memory_type=mtype, importance=imp, metadata={"kind": "note"}
        )
        validate_memory_item(item)
        return await self._manager.store(item)

    @staticmethod
    def classify(text: str) -> MemoryType:
        """Heuristically classify free text into a memory type."""
        low = text.lower()
        if any(cue in low for cue in _PREFERENCE_CUES):
            return MemoryType.PREFERENCE
        if any(cue in low for cue in _DECISION_CUES):
            return MemoryType.DECISION
        if any(cue in low for cue in _TASK_CUES):
            return MemoryType.TASK
        if any(cue in low for cue in _IDEA_CUES):
            return MemoryType.IDEA
        return MemoryType.FACT

    @staticmethod
    def is_meaningful(text: str) -> bool:
        """Return True only for text worth persisting (not greetings/noise)."""
        t = (text or "").strip()
        if len(t) < 8:
            return False
        low = t.lower()
        if any(marker in low for marker in MemoryService._JUNK_RESPONSE_MARKERS):
            return False
        # A bare greeting is not memory-worthy.
        if low.rstrip("!.? ") in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}:
            return False
        return True

    # ==================================================================
    # Internal
    # ==================================================================

    async def _upsert(self, item: MemoryItem) -> None:
        """Replace any existing memory with the same id, then store *item*.

        Forget-then-store guarantees clean replacement regardless of the vector
        backend's duplicate-id semantics (Chroma add vs upsert).
        """
        try:
            await self._manager.forget(item.id)
        except Exception:  # noqa: BLE001 - a missing prior row is fine
            _logger.debug("Upsert: no prior memory for id '%s'", item.id)
        await self._manager.store(item)
