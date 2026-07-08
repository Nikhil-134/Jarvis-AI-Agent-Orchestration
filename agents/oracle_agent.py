"""Oracle agent — knowledge management."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_KNOWLEDGE
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider
from memory import MemoryService

_logger = logging.getLogger(__name__)


class OracleAgent(Agent):
    """Agent responsible for knowledge storage, query, and search."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: Any | None = None,
    ) -> None:
        super().__init__(
            name="oracle",
            supported_task_types=("knowledge.store", "knowledge.query", "knowledge.index", "knowledge.search"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Any]:
        return [CAPABILITY_KNOWLEDGE]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"OracleAgent cannot handle task type: {task.task_type}",
            )

        match task.task_type:
            case "knowledge.store":
                return await self._store(task)
            case "knowledge.query":
                return await self._query(task)
            case "knowledge.index":
                return await self._index(task)
            case "knowledge.search":
                return await self._search(task)
            case _:
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message=f"Unknown task type: {task.task_type}",
                )

    async def _store(self, task: AgentTask) -> AgentResult:
        content = task.payload.get("content", "")
        key = task.payload.get("key")
        _logger.info("Storing knowledge: key=%s", key)
        if self._memory_service is not None:
            try:
                await self._memory_service.store(str(key), content)
            except Exception:
                _logger.exception("Failed to store knowledge")
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message="Failed to store knowledge",
                )
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Knowledge stored successfully.",
            data={"key": key},
        )

    async def _query(self, task: AgentTask) -> AgentResult:
        query = task.payload.get("query", "")
        _logger.info("Querying knowledge: query=%s", query)
        results: list[str] = []
        if self._memory_service is not None:
            try:
                _, items = await self._memory_service.enrich_prompt(query, top_k=5)
                results = [m.content for m in items]
            except Exception:
                _logger.exception("Failed to query knowledge")
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message="Failed to query knowledge",
                )
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Knowledge query completed.",
            data={"results": results, "count": len(results)},
        )

    async def _index(self, task: AgentTask) -> AgentResult:
        content = task.payload.get("content", "")
        source = task.payload.get("source", "unknown")
        _logger.info("Indexing content from source=%s", source)
        if self._memory_service is not None:
            try:
                await self._memory_service.store(f"index:{source}", content)
            except Exception:
                _logger.exception("Failed to index content")
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message="Failed to index content",
                )
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Content indexed successfully.",
            data={"source": source},
        )

    async def _search(self, task: AgentTask) -> AgentResult:
        query = task.payload.get("query", "")
        _logger.info("Searching knowledge base: query=%s", query)
        results: list[str] = []
        if self._memory_service is not None:
            try:
                _, items = await self._memory_service.enrich_prompt(query, top_k=10)
                results = [m.content for m in items]
            except Exception:
                _logger.exception("Failed to search knowledge base")
                return AgentResult(
                    agent_name=self.name,
                    task_id=task.task_id,
                    success=False,
                    message="Failed to search knowledge base",
                )
        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Semantic search completed.",
            data={"results": results, "count": len(results)},
        )
