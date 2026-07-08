"""Friday agent — Jarvis's Research & Information Synthesis specialist."""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.capabilities import CAPABILITY_RESEARCH, Capability
from agents.contracts import AgentResult, AgentTask
from llm.base import BaseLLMProvider
from memory.memory_service import MemoryService
from memory.models import MemoryItem
from tools.engine import ToolExecutionEngine

_logger = logging.getLogger(__name__)


class FridayAgent(Agent):
    """Agent responsible for research, information retrieval, and synthesis."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: ToolExecutionEngine | None = None,
    ) -> None:
        super().__init__(
            name="friday",
            supported_task_types=("research", "information.retrieve", "information.synthesize"),
        )
        self._llm_provider = llm_provider
        self._memory_service = memory_service
        self._tool_engine = tool_engine

    @property
    def capabilities(self) -> list[Capability]:
        return [CAPABILITY_RESEARCH]

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"FridayAgent cannot handle task type: {task.task_type}",
            )

        try:
            if task.task_type == "research":
                return await self._handle_research(task)
            if task.task_type == "information.retrieve":
                return await self._handle_retrieve(task)
            if task.task_type == "information.synthesize":
                return await self._handle_synthesize(task)

            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"Unknown task type: {task.task_type}",
            )
        except Exception as exc:
            _logger.exception("FridayAgent failed for task %s", task.task_id)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"FridayAgent operation failed: {exc}",
                data={"status": "error"},
            )

    async def _handle_research(self, task: AgentTask) -> AgentResult:
        topic = str(task.payload.get("topic", task.payload.get("goal", "")))
        if not topic:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No topic provided for research.",
                data={"status": "error"},
            )

        memories: list[MemoryItem] = []
        if self._memory_service is not None:
            try:
                _, memories = await self._memory_service.enrich_prompt(
                    topic, top_k=5, per_memory_chars=2000, max_context_length=5000,
                )
                _logger.debug("Retrieved %d memories for research on '%s'", len(memories), topic)
            except Exception:
                _logger.exception("Memory retrieval failed during research")

        memory_context = self._format_memory_context(memories) if memories else ""
        response: str = ""

        if self._llm_provider is not None:
            system_prompt = "You are Friday, a research and information synthesis specialist. Provide thorough, well-structured answers."
            if memory_context:
                prompt = f"Research the following topic using the context provided.\n\nTopic: {topic}\n\nRelevant context:\n{memory_context}"
            else:
                prompt = f"Research the following topic thoroughly.\n\nTopic: {topic}"
            try:
                response = await self._llm_provider.generate_text(prompt, system_prompt=system_prompt)
            except Exception:
                _logger.exception("LLM research generation failed")
                response = f"Research on '{topic}' could not be completed due to an LLM error."
        else:
            response = f"Research findings on '{topic}'." + (
                f"\n\nBased on remembered context:\n" + "\n".join(f"- {m.content[:300]}" for m in memories[:3])
                if memories else ""
            )

        if self._memory_service is not None:
            try:
                await self._memory_service.store_interaction(topic, response)
            except Exception:
                _logger.exception("Failed to store research interaction in memory")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Research completed.",
            data={
                "status": "completed",
                "topic": topic,
                "response": response,
                "memory_count": len(memories),
            },
        )

    async def _handle_retrieve(self, task: AgentTask) -> AgentResult:
        query = str(task.payload.get("query", ""))
        top_k = int(task.payload.get("top_k", 5))

        if not query:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Query is required for information retrieval.",
                data={"status": "error"},
            )

        if self._memory_service is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="MemoryService is not configured.",
                data={"status": "unavailable"},
            )

        try:
            items = await self._memory_service.manager.search(query, top_k=top_k)
        except Exception:
            _logger.exception("Semantic search failed for query '%s'", query)
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="Semantic search failed.",
                data={"status": "error"},
            )

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message=f"Found {len(items)} result(s).",
            data={
                "status": "completed",
                "query": query,
                "results": [item.to_document() for item in items],
                "count": len(items),
            },
        )

    async def _handle_synthesize(self, task: AgentTask) -> AgentResult:
        pieces = task.payload.get("pieces", task.payload.get("information", []))
        if isinstance(pieces, str):
            pieces = [pieces]

        if not pieces:
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message="No information pieces provided for synthesis.",
                data={"status": "error"},
            )

        context = "\n\n".join(
            f"Source {i + 1}:\n{piece}" for i, piece in enumerate(pieces)
        )

        summary: str = ""

        if self._llm_provider is not None:
            prompt = (
                "Synthesize the following pieces of information into a coherent, well-structured summary. "
                "Identify common themes, reconcile contradictions, and present a unified picture.\n\n"
                f"{context}"
            )
            try:
                summary = await self._llm_provider.generate_text(
                    prompt,
                    system_prompt="You are Friday, an information synthesis specialist.",
                )
            except Exception:
                _logger.exception("LLM synthesis failed")
                summary = self._fallback_synthesis(pieces)
        else:
            summary = self._fallback_synthesis(pieces)

        if self._memory_service is not None:
            try:
                await self._memory_service.store_fact(
                    f"Synthesized summary: {summary[:500]}", importance=0.6,
                )
            except Exception:
                _logger.debug("Failed to store synthesis in memory")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Synthesis completed.",
            data={
                "status": "completed",
                "source_count": len(pieces),
                "summary": summary,
            },
        )

    async def health_check(self) -> dict[str, object]:
        base = await super().health_check()
        base["llm_available"] = self._llm_provider is not None
        base["memory_available"] = self._memory_service is not None
        base["tool_engine_available"] = self._tool_engine is not None
        capabilities = self.capabilities
        base["capabilities"] = [c.name for c in capabilities]
        return base

    @staticmethod
    def _format_memory_context(memories: list[MemoryItem]) -> str:
        lines = ["Relevant context from memory:"]
        for m in memories:
            lines.append(f"  [{m.memory_type.value}] {m.content[:2000]}")
        return "\n".join(lines)

    @staticmethod
    def _fallback_synthesis(pieces: list[str]) -> str:
        lines = ["Synthesized summary:"]
        for i, piece in enumerate(pieces):
            truncated = piece[:300] + ("..." if len(piece) > 300 else "")
            lines.append(f"  {i + 1}. {truncated}")
        return "\n".join(lines)
