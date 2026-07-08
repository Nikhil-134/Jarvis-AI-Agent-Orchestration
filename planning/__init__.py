"""Planning & Task Execution subsystem (Cycle 8).

An enterprise-grade layer that decomposes complex, actionable goals into a
dependency-aware task graph, executes it concurrently with retries/timeouts/
cancellation, routes each task to a real backend by capability (confidence-based
routing that supersedes regex for actionable goals while regex stays as the
runtime fallback), and verifies the final response before it reaches the user.

Public surface
--------------
* :func:`build_planning_subsystem` — DI factory returning a
  :class:`PlanningCoordinator` (or ``None`` when disabled). This is what
  ``main.py`` calls and stashes on ``orchestrator.planning_coordinator``.
* :class:`PlanningCoordinator` — the single façade the runtime invokes.
* Models + interfaces for typing and testing.

Everything is fail-safe: when disabled or under-provisioned the factory returns
``None`` and the runtime behaves exactly as before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from planning.capabilities import CapabilityCatalog, ResolvedCapability
from planning.coordinator import PlanningCoordinator
from planning.executor import TaskExecutor
from planning.interfaces import (
    ICapabilityCatalog,
    IResponseVerifier,
    ITaskExecutor,
    ITaskPlanner,
    IToolInvoker,
)
from planning.models import (
    ExecutionMetrics,
    NodeResult,
    Plan,
    PlanningOutcome,
    RetryPolicy,
    TaskMetrics,
    TaskNode,
    TaskStatus,
    VerificationResult,
)
from planning.planner import TaskPlanner
from planning.scratchpad import ReasoningScratchpad
from planning.task_graph import TaskGraph
from planning.telemetry import (
    ITelemetrySink,
    InMemoryTelemetrySink,
    LoggingTelemetrySink,
    NullTelemetrySink,
    PlanningTelemetry,
    TelemetryEvent,
    TelemetryKind,
    build_telemetry_sink,
)
from planning.tool_invoker import ToolInvoker
from planning.verifier import ResponseVerifier

if TYPE_CHECKING:
    from config.settings import Settings
    from knowledge.internet import InternetKnowledgeService
    from memory import MemoryService
    from runtime.knowledge_engine import KnowledgeEngine
    from runtime.llm_guard import LLMGuard
    from tools.engine import ToolExecutionEngine

_logger = logging.getLogger(__name__)

__all__ = [
    "CapabilityCatalog",
    "ExecutionMetrics",
    "ICapabilityCatalog",
    "IResponseVerifier",
    "ITaskExecutor",
    "ITaskPlanner",
    "IToolInvoker",
    "ITelemetrySink",
    "InMemoryTelemetrySink",
    "LoggingTelemetrySink",
    "NodeResult",
    "NullTelemetrySink",
    "Plan",
    "PlanningCoordinator",
    "PlanningOutcome",
    "PlanningTelemetry",
    "ReasoningScratchpad",
    "ResolvedCapability",
    "ResponseVerifier",
    "RetryPolicy",
    "TaskExecutor",
    "TaskGraph",
    "TaskMetrics",
    "TaskNode",
    "TaskPlanner",
    "TaskStatus",
    "TelemetryEvent",
    "TelemetryKind",
    "ToolInvoker",
    "VerificationResult",
    "build_planning_subsystem",
    "build_telemetry_sink",
]


def build_planning_subsystem(
    settings: "Settings | None" = None,
    *,
    tool_engine: "ToolExecutionEngine | None" = None,
    memory_service: "MemoryService | None" = None,
    internet_service: "InternetKnowledgeService | None" = None,
    knowledge_engine: "KnowledgeEngine | None" = None,
    llm_guard: "LLMGuard | None" = None,
    progress: Any | None = None,
) -> PlanningCoordinator | None:
    """Compose a :class:`PlanningCoordinator` from injected subsystems.

    Returns ``None`` when planning is disabled in *settings*, so callers can do
    ``orchestrator.planning_coordinator = build_planning_subsystem(...)`` and
    the runtime treats a ``None`` as "planning off" (byte-for-byte fallback).
    """
    def _get(name: str, default: Any) -> Any:
        return getattr(settings, name, default) if settings is not None else default

    if not _get("planning_enabled", True):
        _logger.info("Planning subsystem disabled by settings")
        return None

    max_parallel = int(_get("planning_max_parallel", 4))
    task_timeout = float(_get("planning_task_timeout_seconds", 30.0))
    max_retries = int(_get("planning_max_retries", 1))
    confidence_threshold = float(_get("planning_confidence_threshold", 0.55))
    min_goal_confidence = float(_get("planning_min_goal_confidence", 0.5))
    allow_dangerous = bool(_get("tool_auto_approve", False))
    telemetry_enabled = bool(_get("planning_telemetry_enabled", True))

    planner = TaskPlanner(
        llm_guard=llm_guard,
        default_retry=RetryPolicy(max_retries=max_retries),
    )
    verifier = ResponseVerifier(confidence_threshold=confidence_threshold)
    # Structured, local-only observability (JSON lines → existing rotating log).
    # A null sink when disabled makes telemetry a true no-op.
    telemetry = PlanningTelemetry(build_telemetry_sink(telemetry_enabled))

    coordinator = PlanningCoordinator(
        planner=planner,
        verifier=verifier,
        tool_engine=tool_engine,
        memory_service=memory_service,
        internet_service=internet_service,
        knowledge_engine=knowledge_engine,
        llm_guard=llm_guard,
        max_parallel=max_parallel,
        task_timeout_seconds=task_timeout,
        min_goal_confidence=min_goal_confidence,
        allow_dangerous_tools=allow_dangerous,
        scratchpad_ttl_seconds=float(_get("memory_working_memory_ttl_seconds", 3600)),
        progress=progress,
        telemetry=telemetry,
    )
    _logger.info(
        "Planning subsystem built (max_parallel=%d, task_timeout=%.0fs, "
        "reasoning=%s, telemetry=%s)", max_parallel, task_timeout,
        coordinator.reasoning_available, telemetry_enabled,
    )
    return coordinator
