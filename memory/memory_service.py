"""High-level memory service for agent integration.

Provides the RAG pipeline that agents use to:

1. Enrich prompts with relevant memories
2. Store interactions in long-term memory
3. Manage conversation context
"""

from __future__ import annotations

import logging
from typing import Any

from memory.memory_manager import MemoryManager
from memory.models import MemoryItem, MemoryType, calculate_importance

_logger = logging.getLogger(__name__)


class MemoryService:
    """Agent-facing memory service with RAG pipeline integration.

    Usage::

        service = MemoryService(memory_manager)
        enriched, memories = await service.enrich_prompt("What is JARVIS?")
        await service.store_interaction("What is JARVIS?", "It is an AI OS.")
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self._memory_manager = memory_manager

    @property
    def manager(self) -> MemoryManager:
        """Expose the underlying MemoryManager for advanced operations."""
        return self._memory_manager

    async def enrich_prompt(
        self, prompt: str, top_k: int = 5, max_context_length: int = 2000
    ) -> tuple[str, list[MemoryItem]]:
        """Search memory and inject relevant context into *prompt*.

        Returns a tuple of ``(enriched_prompt, retrieved_memories)``.

        The retrieved memories are prepended as context with metadata
        (type, importance) so the LLM can use them.  Context is
        truncated to *max_context_length* characters to avoid prompt
        pollution.
        """
        try:
            memories = await self._memory_manager.search(prompt, top_k=top_k)
        except Exception:
            _logger.exception("Memory search failed during prompt enrichment")
            return prompt, []

        if not memories:
            return prompt, []

        context_lines = ["Relevant context from memory:"]
        char_count = 0

        for m in memories:
            content = m.content[:500]
            tag = m.memory_type.value
            line = f"  [{tag}] (importance: {m.importance:.2f}) {content}"
            char_count += len(line)
            if char_count > max_context_length:
                break
            context_lines.append(line)

        enriched = "\n".join(context_lines) + "\n\n" + prompt
        _logger.debug("Prompt enriched with %d memory items", len(memories))
        return enriched, memories

    async def store_interaction(
        self, query: str, response: str, importance: float | None = None
    ) -> str:
        """Store a query-response interaction in long-term memory.

        The *importance* is auto-calculated if not provided.
        Returns the memory id.
        """
        content = f"Q: {query}\nA: {response}"
        imp = importance if importance is not None else calculate_importance(content)

        item = MemoryItem(
            content=content,
            memory_type=MemoryType.CONVERSATION,
            importance=imp,
            metadata={
                "query": query,
                "response_preview": response[:200],
            },
        )
        memory_id = await self._memory_manager.store(item)
        _logger.debug("Stored interaction memory '%s' (importance=%.2f)", memory_id, imp)
        return memory_id

    async def store_fact(self, fact: str, importance: float | None = None) -> str:
        """Store a factual memory."""
        imp = importance if importance is not None else calculate_importance(fact, MemoryType.FACT)
        item = MemoryItem(content=fact, memory_type=MemoryType.FACT, importance=imp)
        memory_id = await self._memory_manager.store(item)
        _logger.debug("Stored fact memory '%s'", memory_id)
        return memory_id

    async def store_preference(self, preference: str, importance: float = 0.8) -> str:
        """Store a user preference memory."""
        item = MemoryItem(content=preference, memory_type=MemoryType.PREFERENCE, importance=importance)
        memory_id = await self._memory_manager.store(item)
        _logger.debug("Stored preference memory '%s'", memory_id)
        return memory_id

    async def get_conversation_context(self, n: int = 5) -> str:
        """Return recent conversation context as a formatted string.

        Uses working memory for the most recent turns.
        """
        recent = self._memory_manager.working_memory.get_recent(n=n)
        if not recent:
            return ""

        lines = ["Recent conversation:"]
        for m in recent:
            lines.append(f"  {m.content[:300]}")
        return "\n".join(lines)
