"""Runtime — single entry point for the JARVIS AI Operating System.

Replaces the routing logic in main.py with a clean, layered architecture.
All user input flows through:

    PersonalityEngine → ContextManager → IntentEngine → RoutingEngine
    → Orchestrator → ResponseComposer → User

Usage::

    from runtime import Runtime

    runtime = Runtime(orchestrator, settings)
    response = await runtime.run("Hello!")
    print(response)
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import Settings
from llm import build_llm_provider
from orchestrator import Orchestrator
from tools.intent_detector import IntentDetector

from runtime.conversation_runtime import ConversationRuntime
from runtime.fallback_engine import FallbackEngine
from runtime.llm_guard import LLMGuard, GuardConfig

_logger = logging.getLogger(__name__)


class Runtime:
    """Single entry point for the JARVIS AI Operating System.

    Manages initialization, the conversation pipeline, and lifecycle.
    """

    def __init__(
        self,
        orchestrator: Orchestrator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._settings = settings
        self._fallback = FallbackEngine()

        intent_detector = self._build_intent_detector(orchestrator)
        llm_guard = self._build_llm_guard(settings)
        memory_service = self._extract_memory_service(orchestrator)
        persistent_memory = self._extract_persistent_memory(orchestrator)
        internet_service = self._build_internet_service(orchestrator, settings)
        session_id = getattr(settings, "memory_session_id", "default") if settings else "default"
        internet_max = getattr(settings, "internet_max_results", 5) if settings else 5
        auto_learn = getattr(settings, "memory_auto_learn_enabled", True) if settings else True

        self._conversation = ConversationRuntime(
            orchestrator=orchestrator,
            intent_detector=intent_detector,
            llm_guard=llm_guard,
            memory_service=memory_service,
            persistent_memory=persistent_memory,
            internet_service=internet_service,
            session_id=session_id,
            internet_max_results=internet_max,
            auto_learn_preferences=auto_learn,
        )

        # Planning & Task Execution subsystem (cycle 8). Composed here because it
        # reuses the SAME LLMGuard + KnowledgeEngine the conversation owns (so it
        # shares memory-first context and never double-builds a provider). Wired
        # into the conversation for the actionable-goal path; None → the runtime
        # routes exactly as before (regex fallback only).
        planning_coordinator = self._build_planning_coordinator(
            orchestrator, settings, llm_guard, memory_service, internet_service,
        )
        self._conversation.set_planning_coordinator(planning_coordinator)

    @staticmethod
    def _build_intent_detector(orchestrator: Orchestrator | None) -> IntentDetector | None:
        if orchestrator is None:
            return None
        tool_manager = getattr(orchestrator, "tool_manager", None)
        if tool_manager is None:
            return None
        return IntentDetector(tool_manager)

    @staticmethod
    def _extract_memory_service(orchestrator: Orchestrator | None):
        """Locate the shared MemoryService from the registered jarvis agent."""
        if orchestrator is None:
            return None
        jarvis = orchestrator._agents.get("jarvis")
        return getattr(jarvis, "_memory_service", None) if jarvis else None

    @staticmethod
    def _build_internet_service(orchestrator: Orchestrator | None, settings: Settings | None):
        """Reuse a stashed internet service, else build one from settings.

        Prefers ``orchestrator.internet_service`` (composed once in main.py) so
        the cache/rate-limiter are shared; otherwise constructs the default
        DuckDuckGo+Wikipedia service. Returns None when disabled or on any error
        (fail safe — the pipeline then runs local-only).
        """
        existing = getattr(orchestrator, "internet_service", None)
        if existing is not None:
            return existing
        if settings is not None and not getattr(settings, "internet_enabled", True):
            return None
        try:
            from knowledge.internet import build_internet_service

            return build_internet_service(
                enabled=getattr(settings, "internet_enabled", True) if settings else True,
                timeout=getattr(settings, "internet_timeout_seconds", 6.0) if settings else 6.0,
                overall_timeout=getattr(settings, "internet_overall_timeout_seconds", 8.0) if settings else 8.0,
                cache_ttl_seconds=getattr(settings, "internet_cache_ttl_seconds", 300.0) if settings else 300.0,
                min_interval_seconds=getattr(settings, "internet_min_interval_seconds", 1.0) if settings else 1.0,
            )
        except Exception as exc:  # noqa: BLE001 - never block startup on this
            _logger.warning("Internet knowledge service unavailable: %s", exc)
            return None

    @staticmethod
    def _extract_persistent_memory(orchestrator: Orchestrator | None):
        """Locate the PersistentMemoryService stashed on the orchestrator (if any)."""
        if orchestrator is None:
            return None
        return getattr(orchestrator, "persistent_memory", None)

    def _build_planning_coordinator(
        self,
        orchestrator: Orchestrator | None,
        settings: Settings | None,
        llm_guard: LLMGuard | None,
        memory_service,
        internet_service,
    ):
        """Compose the Planning coordinator, or None (disabled / on any error).

        Reuses the SAME LLMGuard and the conversation's KnowledgeEngine so the
        planner and its reasoning steps share memory-first context. The tool
        engine is taken from the orchestrator (shared registry/permissions).
        Fail-safe: any error logs and yields None → regex-only routing.
        """
        if settings is not None and not getattr(settings, "planning_enabled", True):
            return None
        try:
            from planning import build_planning_subsystem

            tool_engine = getattr(orchestrator, "tool_engine", None)
            knowledge_engine = getattr(self._conversation, "knowledge", None)
            return build_planning_subsystem(
                settings,
                tool_engine=tool_engine,
                memory_service=memory_service,
                internet_service=internet_service,
                knowledge_engine=knowledge_engine,
                llm_guard=llm_guard,
            )
        except Exception as exc:  # noqa: BLE001 - never block startup on planning
            _logger.warning("Planning subsystem unavailable: %s", exc)
            return None

    @staticmethod
    def _build_llm_guard(settings: Settings | None) -> LLMGuard | None:
        if settings is None or not settings.llm_enabled:
            return None
        try:
            primary = build_llm_provider(settings)
            config = GuardConfig(
                primary_timeout_seconds=getattr(settings, "llm_timeout", 30.0),
                max_retries=2,
                retry_backoff_seconds=0.5,
            )
            return LLMGuard(primary, config=config)
        except Exception as exc:
            _logger.warning("Failed to build LLM guard: %s", exc)
            return None

    async def run(self, user_input: str) -> str:
        """Process a single user input and return the response."""
        return await self._conversation.process(user_input)

    async def run_with_metadata(self, user_input: str) -> dict[str, Any]:
        """Process and return response with metadata for diagnostics."""
        response, intent = await self._conversation.process_with_intent(user_input)
        return {
            "response": response,
            "intent": intent.primary.label if intent else "unknown",
            "confidence": intent.primary.confidence if intent else 0.0,
            "context": self._conversation.get_context_summary(),
        }

    @property
    def conversation(self) -> ConversationRuntime:
        return self._conversation

    async def health_check(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "orchestrator": self._orchestrator is not None,
            "session": self._conversation.get_context_summary(),
        }
