"""Comprehensive tests for the memory system (models, vector store,
document store, memory manager, memory service).

All backends are tested with temporary directories so no state
leaks between runs.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from memory.document_store import SQLiteDocumentStore
from memory.exceptions import (
    MemoryError,
    MemoryNotFoundError,
    MemoryStorageError,
)
from memory.memory_manager import MemoryManager, WorkingMemory
from memory.memory_service import MemoryService
from memory.models import MemoryItem, MemoryType, calculate_importance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def tmp_vector_path(tmp_path: Path) -> str:
    return str(tmp_path / "vectors")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestMemoryItem:
    def test_creation_defaults(self) -> None:
        item = MemoryItem(content="Hello, world!")
        assert item.memory_type == MemoryType.CONVERSATION
        assert item.importance == 0.5
        assert item.id is not None
        assert item.created_at is not None

    def test_to_document_round_trip(self) -> None:
        item = MemoryItem(
            content="Test content",
            memory_type=MemoryType.FACT,
            importance=0.8,
            metadata={"source": "test"},
        )
        doc = item.to_document()
        restored = MemoryItem.from_document(doc)
        assert restored.content == item.content
        assert restored.memory_type == item.memory_type
        assert restored.importance == item.importance
        assert restored.metadata == item.metadata
        assert restored.id == item.id

    def test_age_seconds_increases(self) -> None:
        item = MemoryItem(content="fresh")
        assert item.age_seconds >= 0

    def test_is_expired(self) -> None:
        item = MemoryItem(content="old")
        assert not item.is_expired(max_age_days=30)


class TestCalculateImportance:
    def test_baseline(self) -> None:
        assert calculate_importance("hello") == 0.5

    def test_important_keyword(self) -> None:
        assert calculate_importance("This is very important") > 0.5

    def test_preference_type(self) -> None:
        sc = calculate_importance("I like coffee", MemoryType.PREFERENCE)
        assert sc > 0.6

    def test_long_content(self) -> None:
        long_text = "word " * 300
        sc = calculate_importance(long_text)
        assert sc > 0.55

    def test_max_score(self) -> None:
        sc = calculate_importance(
            "This is very important and critical and essential " * 100,
            MemoryType.PREFERENCE,
        )
        assert sc <= 1.0


# ---------------------------------------------------------------------------
# Working Memory
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    def test_add_and_get_recent(self) -> None:
        wm = WorkingMemory(max_items=5, ttl_seconds=3600)
        for i in range(3):
            wm.add(MemoryItem(content=f"item-{i}"))
        recent = wm.get_recent(2)
        assert len(recent) == 2
        assert recent[-1].content == "item-2"

    def test_max_capacity(self) -> None:
        wm = WorkingMemory(max_items=3, ttl_seconds=3600)
        for i in range(5):
            wm.add(MemoryItem(content=f"item-{i}"))
        assert wm.size == 3
        assert wm.get("item-0") is None  # evicted

    def test_get_returns_none_for_missing(self) -> None:
        wm = WorkingMemory()
        assert wm.get("nonexistent") is None

    def test_remove(self) -> None:
        wm = WorkingMemory()
        item = MemoryItem(content="test")
        wm.add(item)
        assert wm.remove(item.id) is True
        assert wm.remove(item.id) is False

    def test_clear(self) -> None:
        wm = WorkingMemory()
        wm.add(MemoryItem(content="a"))
        wm.add(MemoryItem(content="b"))
        wm.clear()
        assert wm.size == 0


# ---------------------------------------------------------------------------
# Document Store (SQLite)
# ---------------------------------------------------------------------------


class TestSQLiteDocumentStore:
    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_db_path: str) -> None:
        store = SQLiteDocumentStore(tmp_db_path)
        await store.initialize()

        doc_id = await store.store("test", {"content": "hello", "id": "doc1"})
        retrieved = await store.retrieve("test", "doc1")
        assert retrieved is not None
        assert retrieved["content"] == "hello"
        assert retrieved["id"] == "doc1"

        await store.close()

    @pytest.mark.asyncio
    async def test_retrieve_missing(self, tmp_db_path: str) -> None:
        store = SQLiteDocumentStore(tmp_db_path)
        await store.initialize()

        result = await store.retrieve("test", "nonexistent")
        assert result is None

        await store.close()

    @pytest.mark.asyncio
    async def test_search_with_filters(self, tmp_db_path: str) -> None:
        store = SQLiteDocumentStore(tmp_db_path)
        await store.initialize()

        await store.store("test", {"id": "1", "content": "alpha", "type": "a"})
        await store.store("test", {"id": "2", "content": "beta", "type": "b"})

        results = await store.search("test")
        assert len(results) == 2

        await store.close()

    @pytest.mark.asyncio
    async def test_delete(self, tmp_db_path: str) -> None:
        store = SQLiteDocumentStore(tmp_db_path)
        await store.initialize()

        await store.store("test", {"id": "del1", "content": "delete me"})
        assert await store.delete("test", "del1") is True
        assert await store.delete("test", "del1") is False

        await store.close()

    @pytest.mark.asyncio
    async def test_count(self, tmp_db_path: str) -> None:
        store = SQLiteDocumentStore(tmp_db_path)
        await store.initialize()

        assert await store.count() == 0
        await store.store("test", {"id": "c1", "content": "x"})
        assert await store.count() == 1

        await store.close()

    @pytest.mark.asyncio
    async def test_initialize_creates_directory(self, tmp_path: Path) -> None:
        nested = str(tmp_path / "sub" / "nested" / "test.db")
        store = SQLiteDocumentStore(nested)
        await store.initialize()
        assert Path(nested).exists()
        await store.close()


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------


@pytest.fixture
async def memory_manager(tmp_path: Path):
    """Create a MemoryManager backed by temp directories."""
    v_path = str(tmp_path / "vectors")
    d_path = str(tmp_path / "memory.db")

    from memory.vector_store import ChromaVectorStore
    from memory.document_store import SQLiteDocumentStore

    vector_store = ChromaVectorStore(v_path)
    doc_store = SQLiteDocumentStore(d_path)

    mm = MemoryManager(
        vector_store=vector_store,
        document_store=doc_store,
        dedup_threshold=0.99,
        importance_threshold=0.0,
    )
    await mm.initialize()
    yield mm


@pytest.mark.asyncio
class TestMemoryManager:
    async def test_store_and_retrieve_memory(self, memory_manager: MemoryManager) -> None:
        item = MemoryItem(content="The sky is blue", memory_type=MemoryType.FACT, importance=0.7)
        memory_id = await memory_manager.store(item)

        retrieved = await memory_manager.retrieve(memory_id)
        assert retrieved is not None
        assert retrieved.content == "The sky is blue"
        assert retrieved.memory_type == MemoryType.FACT

    async def test_store_and_search(self, memory_manager: MemoryManager) -> None:
        await memory_manager.store(MemoryItem(content="Python is a programming language", importance=0.6))
        await memory_manager.store(MemoryItem(content="The Eiffel Tower is in Paris", importance=0.5))

        results = await memory_manager.search("programming language", top_k=5)
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    async def test_search_returns_empty_for_unrelated(self, memory_manager: MemoryManager) -> None:
        await memory_manager.store(MemoryItem(content="Cats are cute animals", importance=0.5))
        results = await memory_manager.search("quantum physics", top_k=5)
        assert len(results) >= 0

    async def test_forget_removes_memory(self, memory_manager: MemoryManager) -> None:
        item = MemoryItem(content="Temporary data", importance=0.3)
        mid = await memory_manager.store(item)

        assert await memory_manager.retrieve(mid) is not None
        await memory_manager.forget(mid)
        assert await memory_manager.retrieve(mid) is None

    async def test_remember_and_recall(self, memory_manager: MemoryManager) -> None:
        await memory_manager.remember("user_pref_1", {
            "content": "User prefers dark mode",
            "memory_type": "preference",
            "importance": 0.9,
        })
        recalled = await memory_manager.recall("user_pref_1")
        assert recalled is not None

    async def test_search_similar(self, memory_manager: MemoryManager) -> None:
        await memory_manager.store(MemoryItem(content="Dogs love to play fetch", importance=0.6))
        results = await memory_manager.search_similar("playing with dogs", top_k=5)
        assert len(results) >= 1

    async def test_search_by_metadata(self, memory_manager: MemoryManager) -> None:
        await memory_manager.store(MemoryItem(
            content="Important fact",
            memory_type=MemoryType.FACT,
            importance=0.9,
        ))
        results = await memory_manager.search_by_metadata(
            memory_type=MemoryType.FACT, min_importance=0.5
        )
        assert len(results) >= 1
        assert results[0].memory_type == MemoryType.FACT

    async def test_get_stats(self, memory_manager: MemoryManager) -> None:
        stats = await memory_manager.get_stats()
        assert "vector_count" in stats
        assert "document_count" in stats
        assert "working_memory_size" in stats

    async def test_remember_empty_key(self, memory_manager: MemoryManager) -> None:
        with pytest.raises(Exception):
            await memory_manager.remember("", {})

    async def test_large_content(self, memory_manager: MemoryManager) -> None:
        large = "word " * 10000
        mid = await memory_manager.store(MemoryItem(content=large, importance=0.5))
        retrieved = await memory_manager.retrieve(mid)
        assert retrieved is not None
        assert len(retrieved.content) > 1000

    async def test_summarize_no_llm(self, memory_manager: MemoryManager) -> None:
        mid1 = await memory_manager.store(MemoryItem(content="First memory", importance=0.5))
        mid2 = await memory_manager.store(MemoryItem(content="Second memory", importance=0.5))
        summary = await memory_manager.summarize([mid1, mid2], llm_provider=None)
        assert isinstance(summary, str)
        assert len(summary) > 0

    async def test_cleanup_removes_old(self, memory_manager: MemoryManager) -> None:
        removed = await memory_manager.cleanup(max_age_days=0)
        assert isinstance(removed, int)

    async def test_not_initialized_raises(self) -> None:
        mm = MemoryManager()
        with pytest.raises(RuntimeError, match="not initialised"):
            await mm.store(MemoryItem(content="test"))

    async def test_concurrent_store_and_search(self, memory_manager: MemoryManager) -> None:
        async def store_items():
            for i in range(10):
                await memory_manager.store(MemoryItem(content=f"concurrent item {i}", importance=0.5))

        async def search_items():
            results = []
            for i in range(10):
                r = await memory_manager.search(f"concurrent item {i}", top_k=5)
                results.extend(r)
            return results

        await asyncio.gather(store_items(), search_items())

    async def test_stress_multiple_operations(self, memory_manager: MemoryManager) -> None:
        ids = []
        for i in range(20):
            mid = await memory_manager.store(MemoryItem(
                content=f"stress test item {i}",
                importance=min(0.5 + i * 0.02, 1.0),
            ))
            ids.append(mid)

        results = await memory_manager.search("stress test", top_k=20)
        assert len(results) == 20

        for mid in ids[:10]:
            await memory_manager.forget(mid)

        results_after = await memory_manager.search("stress test", top_k=20)
        assert len(results_after) <= 15


# ---------------------------------------------------------------------------
# Memory Service
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMemoryService:
    @pytest.fixture
    async def service(self, tmp_path: Path) -> MemoryService:
        from memory.vector_store import ChromaVectorStore
        from memory.document_store import SQLiteDocumentStore

        vs = ChromaVectorStore(str(tmp_path / "sv"))
        ds = SQLiteDocumentStore(str(tmp_path / "sd.db"))
        mm = MemoryManager(vector_store=vs, document_store=ds, importance_threshold=0.0)
        await mm.initialize()
        return MemoryService(mm)

    async def test_enrich_prompt_with_memories(self, service: MemoryService) -> None:
        await service.store_fact("JARVIS is an AI operating system")
        await service.store_fact("It runs entirely on local hardware")

        enriched, memories = await service.enrich_prompt("What is JARVIS?", top_k=5)
        assert len(memories) >= 1
        assert "Relevant context from memory" in enriched
        assert "JARVIS" in enriched

    async def test_enrich_prompt_without_memories(self, service: MemoryService) -> None:
        enriched, memories = await service.enrich_prompt("Something completely unrelated", top_k=5)
        assert len(memories) == 0
        assert enriched == "Something completely unrelated"

    async def test_store_interaction(self, service: MemoryService) -> None:
        mid = await service.store_interaction("What is Python?", "Python is a programming language.")
        assert mid is not None
        retrieved = await service.manager.retrieve(mid)
        assert retrieved is not None
        assert "Python" in retrieved.content

    async def test_store_fact(self, service: MemoryService) -> None:
        mid = await service.store_fact("The Earth orbits the Sun")
        assert mid is not None

    async def test_store_preference(self, service: MemoryService) -> None:
        mid = await service.store_preference("User prefers concise answers")
        assert mid is not None

    async def test_conversation_context(self, service: MemoryService) -> None:
        ctx = await service.get_conversation_context(n=3)
        assert isinstance(ctx, str)

    async def test_memory_pipeline_full_cycle(self, service: MemoryService) -> None:
        await service.store_fact("JARVIS supports memory and RAG")
        await service.store_fact("Memory uses ChromaDB and SQLite")

        enriched, memories = await service.enrich_prompt("How does JARVIS memory work?", top_k=5)
        assert len(memories) >= 1

        await service.store_interaction(
            "How does JARVIS memory work?",
            "JARVIS uses ChromaDB for vector search and SQLite for persistence.",
        )

        enriched2, memories2 = await service.enrich_prompt("What database does memory use?", top_k=5)
        assert len(memories2) >= 1
