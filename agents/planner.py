"""Planner agent — separates internal planning from conversational response.

Architecture
------------
All user input flows through the planner agent.  Processing happens in
two phases so that internal reasoning is *never* exposed to the user:

1. **Phase 1 — Internal plan** (:meth:`_plan`)
   A purely rule-based analysis of the user's goal and any memories
   retrieved by the RAG pipeline.  Returns a structured dict used only
   within this class.  No LLM call is involved.

2. **Phase 2 — Response generation** (:meth:`_respond`)
   Converts the plan + goal + memories into a natural conversational
   response.  When an LLM is configured the "responder" prompt template
   is used; otherwise a simple fallback constructs a reply from the
   available memories.

The final response is returned in ``data["response"]`` — the CLI
**must** display this field and **never** ``data["plan"]`` or any
internal structure.
"""

import logging
from typing import Any

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask
from llm import BaseLLMProvider, ChatSession, LLMError, PromptManager
from memory import MemoryService
from memory.models import MemoryItem

_logger = logging.getLogger(__name__)


class PlannerAgent(Agent):
    """Agent responsible for planning tasks with optional LLM and memory support.

    When a :class:`MemoryService` is provided, the prompt is enriched
    with relevant memories via the RAG pipeline, and each interaction
    is stored for future retrieval.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        prompt_manager: PromptManager | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        super().__init__(name="planner", supported_task_types=("plan",))
        self._prompt_manager = prompt_manager or PromptManager()
        self._memory_service = memory_service
        self._chat_session = (
            ChatSession(llm_provider, system_prompt="You are Jarvis, a helpful AI assistant.")
            if llm_provider
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle(self, task: AgentTask) -> AgentResult:
        if not self.can_handle(task):
            return AgentResult(
                agent_name=self.name,
                task_id=task.task_id,
                success=False,
                message=f"PlannerAgent cannot handle task type: {task.task_type}",
            )

        goal = str(task.payload.get("goal", "No goal provided."))

        # Phase 1 — retrieve memories & build internal plan
        memories = await self._retrieve_memories(goal)
        plan = self._plan(goal, memories)

        # Phase 2 — generate conversational response
        response = await self._respond(plan, goal, memories)

        # Persist interaction for future recall
        if self._memory_service is not None:
            try:
                await self._memory_service.store_interaction(goal, response)
            except Exception:
                _logger.exception("Failed to store planning interaction in memory")

        return AgentResult(
            agent_name=self.name,
            task_id=task.task_id,
            success=True,
            message="Planning completed.",
            data={
                "status": "completed",
                "response": response,
                "memory_enriched": len(memories) > 0,
                "memory_count": len(memories),
            },
        )

    # ------------------------------------------------------------------
    # Phase 1 — internal planning (rule-based, NO LLM)
    # ------------------------------------------------------------------

    def _plan(self, goal: str, memories: list[MemoryItem]) -> dict[str, Any]:
        """Analyse *goal* and *memories* into a structured internal plan.

        The returned dict is used **only** by :meth:`_respond` to guide
        response generation.  It is never returned to the caller or
        exposed in :class:`AgentResult`.
        """
        lower = goal.lower()
        return {
            "is_question": (
                "?" in goal
                or any(lower.startswith(w) for w in ("what", "who", "where", "when", "why", "how", "do", "does", "is", "are", "can", "could", "would", "will", "did"))
            ),
            "is_greeting": any(g in lower for g in ("hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening")),
            "has_memories": len(memories) > 0,
            "memory_count": len(memories),
        }

    # ------------------------------------------------------------------
    # Phase 2 — conversational response generation
    # ------------------------------------------------------------------

    async def _respond(
        self, plan: dict[str, Any], goal: str, memories: list[MemoryItem]
    ) -> str:
        """Generate a natural conversational response from *plan* + *goal* + *memories*.

        When an LLM is available the ``responder`` prompt template is
        rendered with the goal and memory context.  Otherwise a simple
        rule-based fallback constructs a reply.
        """
        if self._chat_session is not None:
            memory_context = self._format_memory_context(memories) if memories else "No relevant memories."
            prompt = self._prompt_manager.render(
                "responder",
                goal=goal,
                memory_context=memory_context,
            )
            try:
                return await self._chat_session.send(prompt)
            except LLMError as exc:
                _logger.exception("LLM response generation failed")
                return self._fallback_response(plan, goal, memories)

        return self._fallback_response(plan, goal, memories)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    async def _retrieve_memories(self, goal: str) -> list[MemoryItem]:
        """Retrieve relevant memories for *goal* via the RAG pipeline."""
        if self._memory_service is None:
            return []

        try:
            _, retrieved = await self._memory_service.enrich_prompt(goal, top_k=5)
            _logger.debug("Retrieved %d memories for planning", len(retrieved))
            return retrieved
        except Exception:
            _logger.exception("Memory retrieval failed")
            return []

    @staticmethod
    def _format_memory_context(memories: list[MemoryItem]) -> str:
        """Format a list of memories into a compact context string."""
        if not memories:
            return "No relevant memories."
        lines = ["Relevant context from memory:"]
        for m in memories:
            lines.append(f"  [{m.memory_type.value}] {m.content[:500]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Fallback (no LLM available)
    # ------------------------------------------------------------------

    def _fallback_response(
        self, plan: dict[str, Any], goal: str, memories: list[MemoryItem]
    ) -> str:
        """Construct a simple response without any LLM call."""
        if plan["is_greeting"]:
            return f"Hello! How can I help you today?"
        if plan["is_question"] and memories:
            parts = [m.content[:300] for m in memories[:3]]
            return "Based on what I know:\n" + "\n".join(f"- {p}" for p in parts)
        if memories:
            parts = [m.content[:300] for m in memories[:3]]
            return "Here's what I found:\n" + "\n".join(f"- {p}" for p in parts)
        return f"I received your message. How can I assist you further?"
