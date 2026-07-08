"""Planner agent — separates internal planning from conversational response.

Architecture
-----------
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

   If a :class:`ToolExecutionEngine` is available, tool definitions are
   sent to the LLM along with the prompt.  If the LLM responds with
   tool calls, they are executed and the results are fed back to the
   LLM for a final conversational response.

The final response is returned in ``data["response"]`` — the CLI
**must** display this field and **never** ``data["plan"]`` or any
internal structure.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import Agent
from agents.contracts import AgentResult, AgentTask
from llm import (
    BaseLLMProvider,
    ChatSession,
    LLMError,
    LLMResponse,
    PromptManager,
    ToolCall,
    ToolDefinition,
)
from memory import MemoryService
from memory.models import MemoryItem

_logger = logging.getLogger(__name__)


class PlannerAgent(Agent):
    """Agent responsible for planning tasks with optional LLM, memory, and tool support.

    When a :class:`MemoryService` is provided, the prompt is enriched
    with relevant memories via the RAG pipeline, and each interaction
    is stored for future retrieval.

    When a :class:`ToolExecutionEngine` is provided, the LLM can select
    and invoke tools during response generation.

    Long prompts are automatically chunked and processed in sequence
    with progress indication when the estimated token count exceeds
    the configured context window.
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        prompt_manager: PromptManager | None = None,
        memory_service: MemoryService | None = None,
        tool_engine: Any | None = None,
        *,
        max_context_tokens: int = 4096,
        max_chunk_tokens: int = 2048,
        chars_per_token: float = 3.5,
    ) -> None:
        super().__init__(name="planner", supported_task_types=("plan",))
        self._prompt_manager = prompt_manager or PromptManager()
        self._memory_service = memory_service
        self._tool_engine = tool_engine
        self._chat_session = (
            ChatSession(llm_provider, system_prompt="You are Jarvis, a helpful AI assistant.")
            if llm_provider
            else None
        )
        self._max_context_tokens = max_context_tokens
        self._max_chunk_tokens = max_chunk_tokens
        self._chars_per_token = chars_per_token

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
        rendered with the goal and memory context.  Tool definitions are
        included so the LLM can invoke tools when appropriate.

        If the estimated token count exceeds the context window, the
        goal is automatically chunked and processed in sequence with
        progress indication.

        Otherwise a simple rule-based fallback constructs a reply.
        """
        if self._chat_session is None:
            return self._fallback_response(plan, goal, memories)

        memory_context = self._format_memory_context(memories) if memories else "No relevant memories."

        # Check if chunking is needed
        from prompt import TokenBudget
        budget = TokenBudget(
            max_context_tokens=self._max_context_tokens,
            chars_per_token=self._chars_per_token,
        )
        budget.allocate_system(self._chat_session.system_prompt)
        budget.allocate_memory(memory_context)

        if budget.can_fit(goal):
            # Single-shot path — fits within the context window
            return await self._respond_single(plan, goal, memory_context)

        # Chunked path — goal exceeds context window
        _logger.info(
            "Goal exceeds context budget (goal ~%d tokens, budget %d tokens). "
            "Chunking into parts.",
            len(goal) // int(self._chars_per_token),
            budget.remaining,
        )
        return await self._respond_chunked(goal, memory_context, plan)

    async def _respond_single(
        self,
        plan: dict[str, Any],
        goal: str,
        memory_context: str,
    ) -> str:
        """Handle a single-shot (non-chunked) prompt."""
        prompt = self._prompt_manager.render(
            "responder",
            goal=goal,
            memory_context=memory_context,
        )

        try:
            tool_defs = self._get_tool_definitions()

            response: LLMResponse = await self._chat_session.send(
                prompt, tools=tool_defs or None,
            )

            if response.tool_calls and self._tool_engine is not None:
                return await self._handle_tool_calls(
                    response.tool_calls, goal, prompt, memory_context,
                )

            # Never return None/empty to the caller — fall back to a real reply.
            return response.content or self._fallback_response(plan, goal, [])

        except LLMError as exc:
            _logger.exception("LLM response generation failed")
            return self._fallback_response(plan, goal, [])

    async def _respond_chunked(
        self,
        goal: str,
        memory_context: str,
        plan: dict[str, Any],
    ) -> str:
        """Process *goal* in chunks when it exceeds the context window.

        Each chunk is sent to the LLM separately with accumulated
        context from previous chunks.  Progress is shown for multi-chunk
        operations.
        """
        from prompt import ChunkProcessor
        processor = ChunkProcessor(
            max_chunk_tokens=self._max_chunk_tokens,
            chars_per_token=self._chars_per_token,
        )

        async def _process_chunk(chunk: str, accumulated: str) -> str:
            """Send one chunk to the LLM and return its response."""
            chunk_prompt = self._build_chunk_prompt(
                chunk, memory_context, accumulated,
            )
            try:
                response = await self._chat_session.provider.generate(
                    chunk_prompt,
                    system_prompt=self._chat_session.system_prompt,
                )
                return response.content
            except Exception:
                _logger.exception("Chunk processing failed")
                return ""

        result = await processor.process_chunked(
            goal,
            chunk_callback=_process_chunk,
            show_progress=True,
        )

        # Store the interaction after all chunks are processed
        if result:
            self._chat_session.append_message("user", goal)
            self._chat_session.append_message("assistant", result)

        return result or self._fallback_response(plan, goal, [])

    @staticmethod
    def _build_chunk_prompt(
        chunk: str,
        memory_context: str,
        accumulated: str,
    ) -> str:
        """Build an LLM prompt for one chunk of a long document."""
        parts = [memory_context]
        if accumulated:
            parts.append(f"Previous context:\n{accumulated}")
        parts.append(
            "Continue processing the following content. "
            "Preserve all important details, formatting, and structure."
        )
        parts.append(chunk)
        return "\n\n".join(parts)

    async def _handle_tool_calls(
        self,
        tool_calls: tuple[ToolCall, ...],
        goal: str,
        original_prompt: str,
        memory_context: str,
    ) -> str:
        """Execute tool calls and return a final conversational response.

        Overwrites the assistant's initial (empty-content) message in
        ChatSession history with the final response.
        """
        _logger.info("LLM requested %d tool call(s)", len(tool_calls))

        results: list[str] = []
        for tc in tool_calls:
            _logger.info("Executing tool '%s' with args=%s", tc.name, tc.arguments)
            result = await self._tool_engine.execute(tc.name, **tc.arguments)
            status = "succeeded" if result.success else "failed"
            output = result.output if result.success else (result.error or "unknown error")
            _logger.info("Tool '%s' %s in %.1f ms", tc.name, status, result.execution_time_ms)
            results.append(f"Tool '{tc.name}' {status}:\n{output}")

        # Build follow-up prompt with tool results
        follow_up = self._prompt_manager.render(
            "responder",
            goal=goal,
            memory_context=memory_context,
            tool_results="\n\n".join(results),
        )

        try:
            final: LLMResponse = await self._chat_session.provider.generate(
                follow_up,
                system_prompt=self._chat_session.system_prompt,
            )

            # Replace the empty assistant message with the real response
            final_content = final.content or ""
            self._chat_session.replace_last_assistant(final_content)

            return final_content or self._fallback_response(
                {"is_question": True, "is_greeting": False, "has_memories": False, "memory_count": 0},
                "Tool results:\n" + "\n".join(results),
                [],
            )
        except LLMError as exc:
            _logger.exception("LLM final response after tool calls failed")
            return self._fallback_response(
                {"is_question": True, "is_greeting": False, "has_memories": False, "memory_count": 0},
                "Tool results:\n" + "\n".join(results),
                [],
            )

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def _get_tool_definitions(self) -> list[ToolDefinition]:
        """Return tool definitions from the tool engine, if available."""
        if self._tool_engine is None:
            return []
        specs = self._tool_engine.registry.list_specs()
        return [
            ToolDefinition(name=s.name, description=s.description, parameters=s.parameters)
            for s in specs
        ]

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    async def _retrieve_memories(self, goal: str) -> list[MemoryItem]:
        """Retrieve relevant memories for *goal* via the RAG pipeline."""
        if self._memory_service is None:
            return []

        try:
            # Use a larger per-memory limit and context cap
            _, retrieved = await self._memory_service.enrich_prompt(
                goal, top_k=5,
                per_memory_chars=2000,
                max_context_length=5000,
            )
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
            lines.append(f"  [{m.memory_type.value}] {m.content[:2000]}")
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
