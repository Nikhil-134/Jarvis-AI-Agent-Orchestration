"""PlanningCoordinator — the single façade the runtime calls.

Requirement #7: integrate cleanly with the existing Runtime/Orchestrator
without breaking current APIs.  The coordinator is the *only* surface the
runtime touches; everything else in ``planning/`` is an internal collaborator.

Flow for one goal::

    run(goal, memory_context)
      │
      ├─ CapabilityCatalog (fresh — reflects live availability)
      ├─ TaskPlanner.decompose ─────────────► Plan
      │        └─ decline if empty / below min-goal-confidence
      ├─ fresh ReasoningScratchpad (per-run; never persisted)
      ├─ TaskExecutor.execute(plan) ────────► ExecutionMetrics (+ node results)
      ├─ synthesize a natural answer from the scratchpad (LLM if available)
      ├─ ResponseVerifier.verify ───────────► VerificationResult
      └─ PlanningOutcome(accepted, response, plan, metrics, verification)

``accepted == False`` tells :class:`ConversationRuntime` to fall through to its
existing regex-routing fallback — the planner never dead-ends a request.

Guarantees: **never raises**; **memory-first** (accepts pre-fetched context,
fetches/stores nothing itself); **local-first** (synthesis uses the local model
via the injected reasoning backend); the internet stays gated inside the
invoker.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from planning.capabilities import CapabilityCatalog
from planning.executor import ProgressCallback, TaskExecutor
from planning.interfaces import IResponseVerifier, ITaskPlanner
from planning.models import (
    ExecutionMetrics,
    NodeResult,
    Plan,
    PlanningOutcome,
    TaskStatus,
)
from planning.scratchpad import ReasoningScratchpad
from planning.telemetry import PlanningTelemetry
from planning.tool_invoker import ToolInvoker

if TYPE_CHECKING:
    from knowledge.internet import InternetKnowledgeService
    from memory import MemoryService
    from runtime.knowledge_engine import KnowledgeEngine
    from runtime.llm_guard import LLMGuard
    from tools.engine import ToolExecutionEngine

_logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = (
    "You are Jarvis. Using ONLY the step results provided, write a single, "
    "clear, friendly answer to the user's goal. Do not mention steps, tools, "
    "JSON, or internal machinery. If a step failed, be honest about what you "
    "could and could not do."
)


class PlanningCoordinator:
    """Orchestrates planner → executor → verifier for one goal. Never raises."""

    def __init__(
        self,
        planner: ITaskPlanner,
        verifier: IResponseVerifier,
        *,
        tool_engine: "ToolExecutionEngine | None" = None,
        memory_service: "MemoryService | None" = None,
        internet_service: "InternetKnowledgeService | None" = None,
        knowledge_engine: "KnowledgeEngine | None" = None,
        llm_guard: "LLMGuard | None" = None,
        max_parallel: int = 4,
        task_timeout_seconds: float = 30.0,
        min_goal_confidence: float = 0.5,
        allow_dangerous_tools: bool = False,
        scratchpad_ttl_seconds: float = 3600.0,
        progress: ProgressCallback | None = None,
        telemetry: PlanningTelemetry | None = None,
    ) -> None:
        self._planner = planner
        self._verifier = verifier
        self._tool_engine = tool_engine
        self._memory = memory_service
        self._internet = internet_service
        self._knowledge = knowledge_engine
        self._guard = llm_guard
        self._max_parallel = max_parallel
        self._task_timeout = task_timeout_seconds
        self._min_goal_confidence = min_goal_confidence
        self._allow_dangerous = allow_dangerous_tools
        self._scratchpad_ttl = scratchpad_ttl_seconds
        self._progress = progress
        # Observability side-channel; null facade by default (no-op when off).
        self._telemetry = telemetry or PlanningTelemetry()

    @property
    def reasoning_available(self) -> bool:
        knowledge_up = self._knowledge is not None and self._knowledge.available
        guard_up = self._guard is not None and self._guard.is_available
        return knowledge_up or guard_up

    async def run(self, goal: str, memory_context: str = "") -> PlanningOutcome:
        """Plan, execute, verify, and synthesise an answer for *goal*.

        Emits exactly one ``run_completed`` telemetry event per call — on every
        terminal path (accepted, declined, or internal error) — carrying the
        wall-clock latency and the fallback ``reason``.
        """
        started = time.monotonic()
        try:
            outcome = await self._run(goal, memory_context)
        except Exception:  # noqa: BLE001 - coordinator must never raise
            _logger.exception("PlanningCoordinator.run failed for goal %r", goal[:60])
            outcome = PlanningOutcome(
                accepted=False, response="", reason="coordinator_error",
            )
        self._emit_run_completed(goal, outcome, (time.monotonic() - started) * 1000.0)
        return outcome

    def _emit_run_completed(
        self, goal: str, outcome: PlanningOutcome, wall_time_ms: float,
    ) -> None:
        """Emit the single terminal ``run_completed`` telemetry record."""
        self._telemetry.run_completed(
            goal=goal,
            accepted=outcome.accepted,
            reason=outcome.reason,
            wall_time_ms=wall_time_ms,
            succeeded=outcome.metrics.succeeded if outcome.metrics else 0,
            total=outcome.metrics.total if outcome.metrics else 0,
            verification_confidence=(
                outcome.verification.confidence if outcome.verification else None
            ),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run(self, goal: str, memory_context: str) -> PlanningOutcome:
        text = (goal or "").strip()
        if not text:
            return PlanningOutcome(accepted=False, response="", reason="empty_goal")

        # 1. Build a live capability catalogue.
        catalog = CapabilityCatalog(
            tool_engine=self._tool_engine,
            memory_service=self._memory,
            internet_service=self._internet,
            reasoning_available=self.reasoning_available,
            allow_dangerous=self._allow_dangerous,
        )

        # 2. Decompose (memory-first: pre-fetched context passed in).
        plan = await self._planner.decompose(text, memory_context, catalog)
        if plan.is_empty:
            self._telemetry.plan_declined(
                goal=text, reason="empty_plan", confidence=plan.overall_confidence,
            )
            return PlanningOutcome(accepted=False, response="", plan=plan,
                                   reason="empty_plan")
        if plan.overall_confidence < self._min_goal_confidence:
            _logger.info("Plan confidence %.2f < %.2f → declining (fallback)",
                         plan.overall_confidence, self._min_goal_confidence)
            self._telemetry.plan_declined(
                goal=text, reason="low_plan_confidence",
                confidence=plan.overall_confidence,
            )
            return PlanningOutcome(accepted=False, response="", plan=plan,
                                   reason="low_plan_confidence")

        # Plan accepted for execution — record the planner's decision.
        self._telemetry.plan_decided(
            goal=text, strategy=plan.strategy, node_count=len(plan.nodes),
            confidence=plan.overall_confidence,
        )

        # 3. Execute with a fresh, per-run scratchpad (never persisted).
        scratchpad = ReasoningScratchpad(ttl_seconds=self._scratchpad_ttl)
        invoker = ToolInvoker(
            catalog,
            tool_engine=self._tool_engine,
            memory_service=self._memory,
            internet_service=self._internet,
            knowledge_engine=self._knowledge,
            llm_guard=self._guard,
            task_timeout_seconds=self._task_timeout,
        )
        executor = TaskExecutor(
            invoker,
            max_parallel=self._max_parallel,
            task_timeout_seconds=self._task_timeout,
            scratchpad=scratchpad,
            progress=self._progress,
            telemetry=self._telemetry,
        )
        metrics = await executor.execute(plan)
        results = scratchpad.all_results()

        # 4. Synthesise a natural answer from the step results.
        draft = await self._synthesize(text, plan, scratchpad)

        # 5. Verify.
        verification = self._verifier.verify(draft, plan, results)

        # If every node failed and there's nothing to salvage, decline so the
        # runtime tries its regex fallback path instead of showing a bare error.
        if metrics.succeeded == 0 and not verification.ok:
            return PlanningOutcome(
                accepted=False, response=verification.response, plan=plan,
                metrics=metrics, verification=verification, reason="all_failed",
            )

        return PlanningOutcome(
            accepted=True, response=verification.response, plan=plan,
            metrics=metrics, verification=verification, reason="ok",
        )

    async def _synthesize(
        self, goal: str, plan: Plan, scratchpad: ReasoningScratchpad,
    ) -> str:
        """Compose a natural answer from the scratchpad's successful outputs."""
        outputs = scratchpad.successful_outputs()

        # Single successful reasoning step → its output IS the answer.
        if plan.is_multi_step is False and outputs:
            return outputs[0]

        if not outputs:
            # Nothing succeeded — let the verifier's salvage/​fallback handle it.
            return ""

        # Multi-step: ask the local model to fuse the step outputs, if we can.
        if self.reasoning_available:
            fused = await self._llm_fuse(goal, outputs)
            if fused:
                return fused

        # No model available → deterministic concatenation (still useful).
        return "\n\n".join(outputs)

    async def _llm_fuse(self, goal: str, outputs: list[str]) -> str:
        steps_block = "\n\n".join(
            f"Step {i} result:\n{out}" for i, out in enumerate(outputs, start=1)
        )
        prompt = f"Goal: {goal}\n\n{steps_block}\n\nAnswer:"
        try:
            if self._knowledge is not None and self._knowledge.available:
                # KnowledgeEngine carries its own system prompt + memory-first
                # context; feed it the fused request as a single turn.
                return (await self._knowledge.answer(prompt)).strip()
            if self._guard is not None and self._guard.is_available:
                response = await self._guard.generate(
                    prompt, system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
                )
                return (response.content or "").strip()
        except Exception:  # noqa: BLE001 - synthesis must never raise
            _logger.debug("LLM fusion failed", exc_info=True)
        return ""
