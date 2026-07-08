"""Conversation Runtime — the central pipeline that processes every user interaction.

Pipeline:
  1. Personality Engine (greetings, humor, sarcasm, small talk)
  2. Context Manager (pronoun resolution, follow-up tracking)
  3. Intent Engine (multi-intent classification)
  4. Routing Engine (dispatch to correct agents)
  5. LLM Guard (retry/timeout/fallback on all LLM calls)
  6. Tool Executor (safe tool execution, no raw JSON)
  7. Response Composer (markdown, artifact stripping, formatting)
  8. Fallback Engine (graceful degradation at every layer)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agents.response_composer import ResponseComposer as LegacyResponseComposer
from orchestrator import Orchestrator
from tools.intent_detector import IntentDetector

if TYPE_CHECKING:
    from memory import MemoryService, PersistentMemoryService
    from knowledge.internet import InternetKnowledgeService
    from planning import PlanningCoordinator

from runtime.context_manager import ContextManager
from runtime.fallback_engine import FallbackEngine
from runtime.intent_engine import IntentEngine, IntentResult
from runtime.knowledge_engine import KnowledgeEngine
from runtime.llm_guard import LLMGuard
from runtime.personality_engine import PersonalityEngine
from runtime.response_composer import RuntimeResponseComposer
from runtime.response_formatter import ResponseFormatter
from runtime.response_synthesizer import ResponseSynthesizer
from runtime.routing_engine import RoutingEngine
from runtime.tool_executor import ToolExecutor

_logger = logging.getLogger(__name__)


class ConversationRuntime:
    """Central pipeline for processing every user interaction.

    Usage::

        runtime = ConversationRuntime(orchestrator, settings)
        result = await runtime.process("Hello!")
        print(result)
    """

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        intent_detector: IntentDetector | None = None,
        llm_guard: LLMGuard | None = None,
        memory_service: "MemoryService | None" = None,
        persistent_memory: "PersistentMemoryService | None" = None,
        internet_service: "InternetKnowledgeService | None" = None,
        *,
        session_id: str = "default",
        internet_max_results: int = 5,
        auto_learn_preferences: bool = True,
    ) -> None:
        self._orchestrator = orchestrator
        self._session_id = session_id
        # Durable cross-session memory. When present, every meaningful turn is
        # recorded under _session_id; the KnowledgeEngine later recalls it.
        self._persistent = persistent_memory
        # Auto-promote preferences stated in conversation to structured profile
        # entries (roadmap #11). Best-effort; off → turns are still recorded but
        # not distilled into structured preferences (pre-cycle-9 behaviour).
        self._auto_learn_preferences = auto_learn_preferences
        # Planning & Task Execution subsystem (cycle 8). Injected post-construction
        # via set_planning_coordinator (it reuses this runtime's KnowledgeEngine).
        # None → actionable goals route via the existing regex RoutingEngine.
        self._planning: "PlanningCoordinator | None" = None

        self.personality = PersonalityEngine()
        self.context = ContextManager(max_turns_per_session=20)
        self.intent_engine = IntentEngine(intent_detector)
        self.routing = RoutingEngine(orchestrator)
        self.llm_guard = llm_guard
        self.tool_executor = ToolExecutor(orchestrator)
        self.synthesizer = ResponseSynthesizer()
        self.formatter = ResponseFormatter()
        self.fallback = FallbackEngine()

        # Direct LLM path for knowledge questions and open chat. Bypasses the
        # rule-based planner and specialist workflow entirely, which is what
        # made general questions return empty / greeting-fallback responses.
        self.knowledge = KnowledgeEngine(
            llm_guard, memory_service,
            internet_service=internet_service,
            persistent_memory=persistent_memory,
            internet_max_results=internet_max_results,
        )

        legacy_composer = None
        if orchestrator:
            jarvis = orchestrator._agents.get("jarvis")
            if jarvis and hasattr(jarvis, "_response_composer"):
                legacy_composer = jarvis._response_composer
        self.response_composer = RuntimeResponseComposer(
            legacy_composer=legacy_composer,
            formatter=self.formatter,
            fallback=self.fallback,
        )

    async def process(self, user_input: str) -> str:
        """Process *user_input* through the full runtime pipeline.

        Returns a clean, formatted response string.
        """
        if not user_input or not user_input.strip():
            return ""

        try:
            return await self._pipeline(user_input.strip())
        except Exception as exc:
            _logger.exception("Runtime pipeline failed")
            result = self.fallback.on_exception(exc, original_goal=user_input)
            return result.data.get("response", "I encountered an unexpected issue. Please try again.")

    async def _pipeline(self, user_input: str) -> str:
        """Execute the full processing pipeline."""
        # Step 1: Personality Engine — fast path for conversational patterns
        personality_response = self.personality.process(user_input)
        if personality_response:
            self.context.update_session(
                self._session_id, user_input, personality_response,
                intent_label="conversation",
            )
            return personality_response

        # Step 2: Context Manager — resolve pronouns and follow-up references
        enriched_input = self.context.enrich(self._session_id, user_input)
        if enriched_input != user_input:
            _logger.info("Context enrichment: '%s' → '%s'", user_input[:40], enriched_input[:40])

        # Step 3: Intent Engine — multi-intent classification
        intent = self.intent_engine.classify(enriched_input)

        # Step 3b: Knowledge / open-chat fast path — answer directly via the
        # LLM. This covers general questions, explanations, follow-ups and any
        # low-confidence intent, and deliberately bypasses the specialist
        # workflow (which cannot answer them and previously returned empty or
        # greeting-fallback text).
        if self._is_knowledge_intent(intent) and self.knowledge.available:
            answer = await self.knowledge.answer(enriched_input)
            answer = self.formatter.format(answer)
            self.context.update_session(
                self._session_id, user_input, answer,
                intent_label=intent.primary.label,
                intent_confidence=intent.primary.confidence,
                enriched_goal=enriched_input,
            )
            await self._persist_turn(user_input, answer)
            return answer

        # Step 3c: Planning & Task Execution — for actionable, multi-step or
        # heavy-planning goals, decompose → execute (parallel, retries, timeout,
        # cancel) → verify. Confidence-based routing that supersedes regex for
        # these goals; on decline/low-confidence it falls through to the regex
        # RoutingEngine below (never dead-ends).
        if self._should_plan(intent):
            outcome = await self._run_planning(enriched_input)
            if outcome is not None and outcome.accepted:
                answer = self.formatter.format(outcome.response)
                self.context.update_session(
                    self._session_id, user_input, answer,
                    intent_label=intent.primary.label,
                    intent_confidence=intent.primary.confidence,
                    enriched_goal=enriched_input,
                )
                await self._persist_turn(user_input, answer)
                return answer
            _logger.info("Planning declined (%s); falling back to regex routing",
                         outcome.reason if outcome else "unavailable")

        # Step 4: Routing Engine — dispatch to correct agent pipeline
        agent_result = await self.routing.route(intent, enriched_input)

        # Step 5: Response Synthesizer — convert raw tool output to natural language
        # Check if the result contains tool output metadata from ToolAgent
        tool_name = (agent_result.data or {}).get("tool_name", "")
        tool_output = (agent_result.data or {}).get("tool_output", "")
        if tool_name and tool_output:
            _logger.debug(
                "Synthesizing tool result: %s (%d chars, success=%s)",
                tool_name, len(tool_output), agent_result.success,
            )
            synthesized = self.synthesizer.synthesize(tool_name, tool_output)
            response = self.formatter.format(synthesized)
        else:
            # Step 5b: Response Composer — format and clean the response
            response = await self.response_composer.compose(
                enriched_input, agent_result,
            )

        # Step 6: Context Manager — store the interaction (short-term, RAM)
        self.context.update_session(
            self._session_id, user_input, response,
            intent_label=intent.primary.label,
            intent_confidence=intent.primary.confidence,
            enriched_goal=enriched_input,
        )

        # Step 7: Persistent memory — durable cross-session record of the turn.
        await self._persist_turn(user_input, response)

        return response

    async def _persist_turn(self, user_input: str, response: str) -> None:
        """Durably record a turn (best-effort; failures are logged, never hidden).

        The persistent layer applies its own store-worthiness gate, so junk /
        empty / tool-JSON turns are skipped there. A storage error must not
        break the reply, but it must be visible in the logs — per the project's
        no-hidden-failures rule.
        """
        if self._persistent is None:
            return
        try:
            await self._persistent.record_turn(self._session_id, user_input, response)
        except Exception:  # noqa: BLE001 - never let a memory write break the chat
            _logger.exception(
                "Persistent record_turn failed for session '%s'", self._session_id
            )
        # Auto-promote any preference stated in this turn to the structured
        # profile (roadmap #11). Runs independently of record_turn's
        # store-worthiness gate — a short "call me Boss" is worth learning even
        # if the turn itself is borderline. getattr-guarded so a minimal
        # persistent-memory fake (record_turn only) is unaffected.
        if not self._auto_learn_preferences:
            return
        learn = getattr(self._persistent, "learn_preferences", None)
        if learn is None:
            return
        try:
            await learn(user_input)
        except Exception:  # noqa: BLE001 - preference learning is best-effort
            _logger.debug("Auto-learn preferences failed", exc_info=True)

    # Intents that have real, dedicated handlers and must NOT be diverted to
    # the direct-LLM knowledge path.
    _SPECIALIST_LABELS = frozenset({
        "tool", "coding", "security", "devops", "shell",
        "browser", "desktop", "calendar", "reminder", "email", "notes",
    })

    def _is_knowledge_intent(self, intent: IntentResult) -> bool:
        """Return True if *intent* should be answered by the KnowledgeEngine.

        Knowledge questions, chit-chat, follow-ups, and any low-confidence or
        unrecognised intent go to the KnowledgeEngine. Intents with concrete
        specialist/tool handlers are left for the routing engine.

        ``current_info`` (weather / news / latest / current office-holders) is
        deliberately a KnowledgeEngine intent: that path — and ONLY that path —
        consults the router-gated :class:`InternetKnowledgeService` for live
        facts, keeping reasoning local. Routing it to the browser stub instead
        would strand these time-sensitive queries with no live data.
        """
        label = intent.primary.label
        # current_info must reach the KnowledgeEngine (→ InternetKnowledgeService),
        # even though the classifier also flags requires_browser for it.
        if label == "current_info":
            return True
        if label in self._SPECIALIST_LABELS:
            return False
        if intent.requires_tool or intent.requires_vision or intent.requires_browser:
            return False
        # knowledge_question, follow_up, unknown, greeting/conversation that
        # slipped past the personality engine, plan, or anything low-confidence.
        return True

    # ------------------------------------------------------------------
    # Planning & Task Execution integration (cycle 8)
    # ------------------------------------------------------------------

    def set_planning_coordinator(self, coordinator: "PlanningCoordinator | None") -> None:
        """Inject the Planning coordinator (composed by the Runtime).

        Kept as a post-construction setter because the coordinator reuses this
        runtime's KnowledgeEngine — a constructor argument would create a
        chicken-and-egg dependency.
        """
        self._planning = coordinator

    def _should_plan(self, intent: IntentResult) -> bool:
        """Return True if *intent* should go through the Planning coordinator.

        Scoped deliberately narrow (per design): only genuinely *actionable* and
        *complex* goals — multi-step (compound) requests or explicit
        heavy-planning intents. Plain chat, knowledge questions, greetings,
        single-tool requests, ``current_info`` (which must reach the internet
        via the KnowledgeEngine), and vision all stay on their existing paths.
        This preserves the stabilized memory-first / local-first behaviour.
        """
        if self._planning is None:
            return False
        if not intent.is_actionable:
            return False
        # current_info must stay on the KnowledgeEngine → InternetKnowledgeService
        # path; and pure conversation/vision have their own handlers.
        if intent.primary.label == "current_info":
            return False
        if intent.requires_conversation or intent.requires_vision:
            return False
        # Complex = multi-step (compound classify populated `secondary`) OR an
        # explicit heavy-planning intent. Single-tool requests (`requires_tool`)
        # stay on the proven regex fast-path.
        is_multi_step = len(intent.secondary) >= 1
        return is_multi_step or intent.requires_planning

    async def _run_planning(self, enriched_input: str):
        """Run the Planning coordinator with memory-first context. Never raises.

        Returns the PlanningOutcome, or None if planning is unavailable/errored
        (caller then falls through to regex routing).
        """
        if self._planning is None:
            return None
        try:
            memory_context = await self._gather_memory_context(enriched_input)
            return await self._planning.run(enriched_input, memory_context)
        except Exception:  # noqa: BLE001 - planning must never break the pipeline
            _logger.exception("Planning coordinator failed; will fall back")
            return None

    async def _gather_memory_context(self, query: str) -> str:
        """Pre-fetch local memory context (memory-first) to hand to the planner.

        Reuses the KnowledgeEngine's own read-only recall so the planner and the
        chat path see the same local memories. Best-effort: returns "" on any
        issue. The planner fetches nothing itself — single reader discipline.
        """
        knowledge = getattr(self, "knowledge", None)
        retriever = getattr(knowledge, "_retrieve_memory_context", None)
        if retriever is None:
            return ""
        try:
            return await retriever(query)
        except Exception:  # noqa: BLE001
            _logger.debug("Memory pre-fetch for planning failed", exc_info=True)
            return ""

    async def process_with_intent(self, user_input: str) -> tuple[str, IntentResult]:
        """Process and return both the response and the intent classification.

        Useful for debugging and logging.
        """
        enriched = self.context.enrich(self._session_id, user_input)
        intent = self.intent_engine.classify(enriched)
        response = await self.process(user_input)
        return response, intent

    def set_session(self, session_id: str) -> None:
        """Switch to a different user session."""
        self._session_id = session_id

    def clear_session(self) -> None:
        """Clear all context for the current session."""
        self.context.clear_session(self._session_id)

    def get_context_summary(self) -> str:
        """Return a summary of the current session context."""
        return self.context.get_context_summary(self._session_id)
