"""Knowledge Engine — direct, reliable LLM conversation for knowledge and
open-ended chat.

Rationale
---------
The legacy knowledge path routed general questions ("explain photosynthesis",
"what is the capital of France") through the rule-based ``PlannerAgent`` and
the specialist ``WorkflowEngine``.  That path:

* injected all 14 tool definitions into every prompt, causing small local
  models (e.g. ``qwen2.5-coder:3b``) to emit tool-call JSON instead of prose;
* fed retrieved memories — including *failed* prior turns — back into the
  prompt, creating a self-reinforcing degradation loop;
* fell back to ``"Hello! How can I help you today?"`` whenever any of the
  fragile heuristics misfired.

``KnowledgeEngine`` replaces that path with the simplest thing that works:
a direct LLM call carrying real conversation history and *optional*,
read-only long-term memory as context.  No planner, no workflow, no tool
definitions.  All calls go through :class:`~runtime.llm_guard.LLMGuard`, so
timeouts and provider errors degrade gracefully instead of crashing.

Memory here is *read-only*: the engine may pull relevant facts in, but it
never writes low-value turns back — storage decisions live in
:class:`~memory.memory_service.MemoryService`, which now gates junk out.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from knowledge.internet.router import is_memory_query
from memory import MemoryService
from runtime.llm_guard import LLMGuard

if TYPE_CHECKING:
    from knowledge.internet import InternetKnowledgeService
    from memory import PersistentMemoryService

_logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are Jarvis, a helpful, friendly and knowledgeable AI assistant. "
    "Answer the user directly and conversationally. Be concise unless the "
    "user asks for detail. You may use light humour and acknowledge jokes or "
    "sarcasm naturally. If you genuinely do not know something, say so briefly "
    "rather than inventing facts. Never mention tools, JSON, prompts, or your "
    "internal machinery — the user only sees your final answer."
)

# How many prior turns to carry as conversation history.
_MAX_HISTORY_TURNS = 8


@dataclass(frozen=True, slots=True)
class _Turn:
    user: str
    assistant: str


class KnowledgeEngine:
    """Answer knowledge questions and open chat directly via the LLM.

    Usage::

        engine = KnowledgeEngine(llm_guard, memory_service)
        answer = await engine.answer("Explain how photosynthesis works")
    """

    def __init__(
        self,
        llm_guard: LLMGuard | None,
        memory_service: MemoryService | None = None,
        *,
        internet_service: "InternetKnowledgeService | None" = None,
        persistent_memory: "PersistentMemoryService | None" = None,
        max_history_turns: int = _MAX_HISTORY_TURNS,
        use_memory_context: bool = True,
        internet_max_results: int = 5,
    ) -> None:
        self._guard = llm_guard
        self._memory = memory_service
        # Durable cross-session memory (profile/preferences + semantic recall).
        # Consulted BEFORE the LLM for personal/identity questions so answers
        # survive a restart. Never sent to the internet.
        self._persistent = persistent_memory
        # Internet retrieval is a *last-resort context source*: consulted only
        # when the freshness router says local memory + the local model cannot
        # answer (weather/news/current events). Never stores; reasoning stays
        # local. Injected as untrusted context into the prompt only.
        self._internet = internet_service
        self._internet_max_results = internet_max_results
        self._history: deque[_Turn] = deque(maxlen=max_history_turns)
        self._use_memory_context = use_memory_context

    @property
    def available(self) -> bool:
        """Whether a usable LLM guard is configured."""
        return self._guard is not None and self._guard.is_available

    async def answer(self, user_input: str) -> str:
        """Return a natural-language answer to *user_input*.

        Never raises: on any failure returns a short, honest fallback so the
        conversation loop keeps running.
        """
        text = (user_input or "").strip()
        if not text:
            return ""

        if not self.available:
            return (
                "I can't reach a language model right now, so I can't answer "
                "that. Please make sure Ollama is running and try again."
            )

        # Priority ladder (steps 1–4): local memory / conversation first.
        memory_context = await self._retrieve_memory_context(text)

        # Step 5: internet retrieval — ONLY if the query is time-sensitive and
        # local memory did not already surface relevant context. This keeps the
        # internet strictly a last resort behind local sources.
        internet_context = await self._retrieve_internet_context(
            text, local_context_found=bool(memory_context),
        )

        prompt = self._build_prompt(text, memory_context, internet_context)

        try:
            response = await self._guard.generate(prompt, system_prompt=_SYSTEM_PROMPT)
            answer = (response.content or "").strip()
        except Exception:  # LLMGuard.generate is designed not to raise, belt-and-braces
            _logger.exception("KnowledgeEngine LLM call failed")
            answer = ""

        if not answer:
            return "I'm not sure how to answer that one — could you rephrase it?"

        self._history.append(_Turn(user=text, assistant=answer))
        return answer

    def reset(self) -> None:
        """Clear conversation history (new session)."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _retrieve_memory_context(self, query: str) -> str:
        """Pull relevant long-term memories as read-only context (best effort).

        Priority ladder (steps 1–2 of the project's fixed order): durable
        persistent memory and semantic recall are consulted *before* the LLM.
        Personal/identity questions ("who am I", "what do I like") recall a
        deeper set so a stored fact reliably surfaces across a restart. If
        nothing is remembered we return "" and the model answers naturally.
        """
        if not self._use_memory_context:
            return ""

        is_personal = is_memory_query(query)
        lines: list[str] = []

        # 1. Structured user profile (preferences) — cheap dict recall, durable.
        if is_personal and self._persistent is not None:
            try:
                profile = await self._persistent.get_profile()
            except Exception:
                _logger.debug("Profile recall failed in KnowledgeEngine", exc_info=True)
                profile = {}
            for key, value in profile.items():
                if value:
                    lines.append(f"- {key}: {value}")

        # 2. Semantic recall over durable memory (conversation turns, facts).
        if self._memory is not None:
            top_k = 6 if is_personal else 3
            try:
                _, memories = await self._memory.enrich_prompt(
                    query, top_k=top_k, per_memory_chars=500, max_context_length=2000,
                )
            except Exception:
                _logger.debug("Memory retrieval failed in KnowledgeEngine", exc_info=True)
                memories = []
            for m in memories:
                content = (getattr(m, "content", "") or "").strip()
                if content:
                    lines.append(f"- {content[:500]}")

        if not lines:
            return ""
        return "Some things I remember that may be relevant:\n" + "\n".join(lines)

    async def _retrieve_internet_context(
        self, query: str, *, local_context_found: bool
    ) -> str:
        """Fetch live public facts as context — only when genuinely required.

        Privacy: only the *current question* is sent outward — never history,
        memory, secrets, or internal prompts. Fail-safe: any error/timeout
        yields an empty string and we proceed with local knowledge only.
        """
        if self._internet is None or not self._internet.available:
            return ""

        # Late import keeps the router optional and avoids a hard dependency
        # cycle at module import time.
        from knowledge.internet import needs_internet

        if not needs_internet(query, local_context_found=local_context_found):
            return ""

        try:
            context = await self._internet.build_context(
                query, max_results=self._internet_max_results,
            )
        except Exception:  # noqa: BLE001 - retrieval must never break the answer
            _logger.debug("Internet retrieval failed in KnowledgeEngine", exc_info=True)
            return ""

        if context:
            _logger.info("Injected live internet context for query %r", query[:60])
        return context

    def _build_prompt(
        self, user_input: str, memory_context: str, internet_context: str = ""
    ) -> str:
        """Assemble history + memory + live context + current turn into a prompt.

        We render history as plain ``User:``/``Jarvis:`` lines rather than a
        provider-specific messages array so this stays portable across the
        provider-neutral ``generate`` interface.
        """
        parts: list[str] = []
        if memory_context:
            parts.append(memory_context)
            parts.append("")
        if internet_context:
            parts.append(internet_context)
            parts.append("")

        for turn in self._history:
            parts.append(f"User: {turn.user}")
            parts.append(f"Jarvis: {turn.assistant}")

        parts.append(f"User: {user_input}")
        parts.append("Jarvis:")
        return "\n".join(parts)
