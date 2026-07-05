"""Memory manager — orchestrates vector store, document store, embeddings,
and working memory into a unified memory system."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any

from llm.base import BaseLLMProvider
from memory.document_store import SQLiteDocumentStore
from memory.embedding_provider import ChromaEmbeddingProvider
from memory.exceptions import MemoryDeduplicationError, MemoryError
from memory.interfaces import IDocumentStore, IEmbeddingProvider, IMemoryStore, IVectorStore
from memory.models import MemoryItem, MemoryType, calculate_importance
from memory.vector_store import ChromaVectorStore

_logger = logging.getLogger(__name__)


class WorkingMemory:
    """Short-term in-memory context with fixed capacity and TTL eviction.

    Expired-item scans are throttled to once per second to avoid O(n)
    iteration on every operation in high-throughput scenarios.
    """

    def __init__(self, max_items: int = 50, ttl_seconds: int = 3600) -> None:
        self._items: OrderedDict[str, MemoryItem] = OrderedDict()
        self._max_items = max_items
        self._ttl_seconds = ttl_seconds
        self._last_evict: float = 0.0
        self._evict_interval = 1.0

    def add(self, item: MemoryItem) -> None:
        self._evict_expired()
        self._items[item.id] = item
        self._items.move_to_end(item.id)
        while len(self._items) > self._max_items:
            self._items.popitem(last=False)

    def get(self, memory_id: str) -> MemoryItem | None:
        self._evict_expired()
        item = self._items.get(memory_id)
        if item is not None:
            self._items.move_to_end(memory_id)
        return item

    def get_recent(self, n: int = 10) -> list[MemoryItem]:
        self._evict_expired()
        return list(self._items.values())[-n:]

    def remove(self, memory_id: str) -> bool:
        return self._items.pop(memory_id, None) is not None

    def clear(self) -> None:
        self._items.clear()

    @property
    def size(self) -> int:
        self._evict_expired()
        return len(self._items)

    def _evict_expired(self) -> None:
        now = datetime.now(timezone.utc).timestamp()
        if now - self._last_evict < self._evict_interval:
            return
        self._last_evict = now
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._ttl_seconds)
        expired = [mid for mid, item in self._items.items() if item.created_at < cutoff]
        for mid in expired:
            del self._items[mid]


_DEFAULT_DEDUP_THRESHOLD = 0.95
_DEFAULT_IMPORTANCE_THRESHOLD = 0.3


class MemoryManager(IMemoryStore):
    """Unified memory manager combining vector search, document persistence,
    embedding generation, and working memory.

    Flow for every memory operation:
    1. Embed content (if needed)
    2. Check for duplicates
    3. Store in vector store for semantic search
    4. Store in document store for structured retrieval
    5. Keep recent items in working memory for fast access
    """

    def __init__(
        self,
        vector_store: IVectorStore | None = None,
        document_store: IDocumentStore | None = None,
        embedding_provider: IEmbeddingProvider | None = None,
        working_memory: WorkingMemory | None = None,
        dedup_threshold: float = _DEFAULT_DEDUP_THRESHOLD,
        importance_threshold: float = _DEFAULT_IMPORTANCE_THRESHOLD,
    ) -> None:
        self._vector_store = vector_store or ChromaVectorStore(
            path="./memory_data/vectors"
        )
        self._document_store = document_store or SQLiteDocumentStore(
            path="./memory_data/documents.db"
        )
        self._embedding = embedding_provider or ChromaEmbeddingProvider()
        self._working = working_memory or WorkingMemory()
        self._dedup_threshold = dedup_threshold
        self._importance_threshold = importance_threshold
        self._initialized = False

    async def initialize(self) -> None:
        """Initialise all backends (must be called before use)."""
        await self._vector_store.initialize()
        await self._document_store.initialize()
        self._initialized = True
        _logger.info("MemoryManager initialised")

    @property
    def working_memory(self) -> WorkingMemory:
        """Expose working memory for service-layer access (avoids private attr access)."""
        return self._working

    # ------------------------------------------------------------------
    # IMemoryStore implementation
    # ------------------------------------------------------------------

    async def remember(self, key: str, data: dict[str, Any]) -> None:
        """Store a data dict under a string key (simple key-value API).

        The *key* is used as the memory id.  The data dict is wrapped
        in a :class:`MemoryItem` and stored across all backends.
        """
        self._ensure_ready()

        item = MemoryItem(
            id=key,
            content=str(data.get("content", "")),
            memory_type=MemoryType(data.get("memory_type", "fact")),
            importance=data.get("importance", calculate_importance(str(data))),
            metadata={k: v for k, v in data.items() if k not in ("content", "memory_type", "importance")},
        )
        await self.store(item)

    async def recall(self, key: str) -> dict[str, Any] | None:
        """Retrieve a data dict by its key.

        Checks working memory first, then document store.
        """
        self._ensure_ready()

        item = self._working.get(key)
        if item is not None:
            return item.to_document()

        doc = await self._document_store.retrieve("memory", key)
        if doc is None:
            return None
        return doc

    async def search_similar(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search memories semantically similar to *query*."""
        self._ensure_ready()
        memories = await self._search(query, top_k=top_k)
        return [m.to_document() for m in memories]

    async def forget(self, key: str) -> bool:
        """Remove a memory by its id.

        Removes from working memory, document store, and vector store.
        Returns True if at least one backend held the memory.
        """
        self._ensure_ready()

        working_removed = self._working.remove(key)
        doc_deleted = await self._document_store.delete("memory", key)
        if doc_deleted:
            try:
                await self._vector_store.delete([key])
            except Exception:
                _logger.debug("Vector delete failed for '%s' (may not exist)", key)
            return True
        return working_removed

    # ------------------------------------------------------------------
    # Extended API
    # ------------------------------------------------------------------

    async def store(self, item: MemoryItem) -> str:
        """Store a :class:`MemoryItem` across all backends.

        1. Generate embedding (if not provided)
        2. Check for duplicate content
        3. Store vector in ChromaDB
        4. Store document in SQLite
        5. Add to working memory
        """
        self._ensure_ready()

        if item.embedding is None:
            item.embedding = await self._embedding.embed(item.content)

        duplicate_id = await self._check_duplicate(item)
        if duplicate_id is not None:
            _logger.debug("Skipped duplicate memory (similar to '%s')", duplicate_id)
            return duplicate_id

        await self._vector_store.add_vectors(
            vectors=[item.embedding],
            metadata=[item.to_vector_metadata()],
            ids=[item.id],
        )

        await self._document_store.store("memory", item.to_document())

        self._working.add(item)

        _logger.info("Stored memory '%s' (type=%s, importance=%.2f)", item.id, item.memory_type.value, item.importance)
        return item.id

    async def retrieve(self, memory_id: str) -> MemoryItem | None:
        """Retrieve a :class:`MemoryItem` by its id.

        Checks working memory first, then document store.
        """
        self._ensure_ready()

        working = self._working.get(memory_id)
        if working is not None:
            return working

        doc = await self._document_store.retrieve("memory", memory_id)
        if doc is None:
            return None

        item = MemoryItem.from_document(doc)
        item.embedding = await self._embedding.embed(item.content)
        return item

    async def search(
        self, query: str, top_k: int = 10
    ) -> list[MemoryItem]:
        """Semantic search returning :class:`MemoryItem` instances."""
        self._ensure_ready()

        results = await self._search(query, top_k=top_k)
        return results

    async def search_by_metadata(
        self,
        memory_type: MemoryType | None = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[MemoryItem]:
        """Search documents by structured metadata filters."""
        self._ensure_ready()

        filters: dict[str, Any] = {}
        if memory_type is not None:
            filters["memory_type"] = memory_type.value

        docs = await self._document_store.search("memory", filters=filters, limit=limit)

        items = [MemoryItem.from_document(d) for d in docs]
        items = [it for it in items if it.importance >= min_importance]
        items.sort(key=lambda it: it.importance, reverse=True)
        return items[:limit]

    async def summarize(
        self, memory_ids: list[str], llm_provider: BaseLLMProvider | None = None
    ) -> str:
        """Generate a summary of the specified memories.

        If *llm_provider* is provided, uses it for intelligent
        summarisation.  Otherwise returns a concatenation.
        """
        self._ensure_ready()

        memories: list[MemoryItem] = []
        for mid in memory_ids:
            item = await self.retrieve(mid)
            if item is not None:
                memories.append(item)

        if not memories:
            return ""

        if llm_provider is not None:
            combined = "\n".join(f"- {m.content}" for m in memories)
            prompt = f"Summarise the following information concisely:\n\n{combined}"
            try:
                return await llm_provider.generate(prompt, system_prompt="You are a summarisation assistant.")
            except Exception:
                _logger.warning("LLM summarisation failed, falling back to concatenation")

        return " | ".join(m.content for m in memories[:5])

    async def cleanup(self, max_age_days: int = 30) -> int:
        """Remove memories older than *max_age_days*.

        Returns the number of memories removed.
        """
        self._ensure_ready()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        docs = await self._document_store.search("memory", limit=10000)

        old_ids = [doc["id"] for doc in docs if doc.get("created_at", "") < cutoff]

        for mid in old_ids:
            await self.forget(mid)

        if old_ids:
            _logger.info("Cleanup removed %d old memories", len(old_ids))
        return len(old_ids)

    async def get_stats(self) -> dict[str, Any]:
        """Return diagnostic statistics about the memory system."""
        self._ensure_ready()

        vector_count = 0
        doc_count = 0
        try:
            vector_count = await self._vector_store.count()
        except (RuntimeError, MemoryError):
            pass
        try:
            doc_count = await self._document_store.count("memory")
        except (RuntimeError, MemoryError):
            pass

        return {
            "vector_count": vector_count,
            "document_count": doc_count,
            "working_memory_size": self._working.size,
            "importance_threshold": self._importance_threshold,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _search(
        self, query: str, top_k: int = 10
    ) -> list[MemoryItem]:
        """Execute a semantic search and return ranked MemoryItems."""
        query_embedding = await self._embedding.embed(query)

        results = await self._vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
        )

        items: list[MemoryItem] = []
        for r in results:
            item = MemoryItem.from_vector_metadata(
                r,
                embedding=r.get("embedding"),
                distance=1.0 - r.get("score", 0.0),
            )
            if item.importance >= self._importance_threshold:
                items.append(item)

        return items

    async def _check_duplicate(self, item: MemoryItem) -> str | None:
        """Return the id of an existing memory whose content is a near-duplicate.

        Uses cosine similarity against the vector store.
        """
        try:
            results = await self._vector_store.search(
                query_vector=item.embedding,
                top_k=1,
            )
            if results and results[0].get("score", 0) >= self._dedup_threshold:
                return results[0]["id"]
        except Exception as exc:
            raise MemoryDeduplicationError(f"Deduplication check failed: {exc}") from exc
        return None

    def _ensure_ready(self) -> None:
        if not self._initialized:
            raise RuntimeError("MemoryManager not initialised. Call .initialize() first.")
