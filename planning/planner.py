"""TaskPlanner — decomposes a complex goal into an executable :class:`Plan`.

This is the "PlannerAgent" role of the new subsystem (requirement #1).  It is
**distinct** from the legacy :class:`agents.planner.PlannerAgent` (a
conversational responder registered under task_type ``"plan"``): the
``TaskPlanner`` is a plain injected collaborator inside the coordinator, is
never registered with the orchestrator, and never uses the ``"plan"`` task
type.

Strategy (mirrors :class:`memory.reflection.ReflectionEngine`)
--------------------------------------------------------------
1. Ask the local model (via :class:`LLMGuard`, which never raises) for a strict
   JSON decomposition, giving it the *available* capabilities and the
   already-retrieved memory context (memory-first — the planner fetches
   nothing itself).
2. Parse the JSON defensively.  On any parse failure — including the graceful
   fallback string the guard returns when Ollama is down — fall back to a
   **deterministic heuristic** decomposition so planning still works offline.

The planner writes nothing to memory and performs no tool calls; it only
produces a plan for the executor.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from planning.capabilities import CapabilityCatalog
from planning.interfaces import ICapabilityCatalog, ITaskPlanner
from planning.models import Plan, RetryPolicy, TaskNode

try:  # LLMGuard is optional at construction time
    from runtime.llm_guard import LLMGuard
except Exception:  # pragma: no cover - defensive
    LLMGuard = None  # type: ignore[assignment,misc]

_logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = (
    "You are Jarvis's task planner. Decompose the user's goal into the minimum "
    "set of concrete steps. Respond with STRICT JSON only — no prose, no code "
    "fences. Schema:\n"
    '{"steps": [{"id": "s1", "description": "...", "tool": "<capability>", '
    '"depends_on": ["s0"], "confidence": 0.0-1.0, "cost": 1-5}], '
    '"confidence": 0.0-1.0}\n'
    "Rules: use ONLY the listed capabilities for \"tool\"; keep ids short and "
    "unique; \"depends_on\" lists ids of steps that must finish first; a step "
    "that just answers/explains uses tool \"reasoning\"; prefer 1-4 steps."
)

# Split a compound goal on coordinating conjunctions for the heuristic path.
_STEP_SPLITTER = re.compile(
    r"\s*(?:,?\s*and\s+then\s+|\s+then\s+|,?\s*and\s+also\s+|;\s*|,\s+)",
    re.IGNORECASE,
)

_MAX_STEPS = 8


class TaskPlanner(ITaskPlanner):
    """Turns a goal + capabilities + memory context into a :class:`Plan`."""

    def __init__(
        self,
        llm_guard: "LLMGuard | None" = None,
        *,
        max_steps: int = _MAX_STEPS,
        default_retry: RetryPolicy | None = None,
    ) -> None:
        self._guard = llm_guard
        self._max_steps = max_steps
        self._default_retry = default_retry or RetryPolicy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def decompose(
        self,
        goal: str,
        memory_context: str,
        catalog: ICapabilityCatalog,
    ) -> Plan:
        """Return a :class:`Plan` for *goal*. Never raises."""
        text = (goal or "").strip()
        if not text:
            return Plan(goal="", nodes=(), overall_confidence=0.0, strategy="single")

        llm_plan = await self._plan_with_llm(text, memory_context, catalog)
        if llm_plan is not None and not llm_plan.is_empty:
            return llm_plan

        return self._plan_heuristic(text, catalog)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    async def _plan_with_llm(
        self, goal: str, memory_context: str, catalog: ICapabilityCatalog,
    ) -> Plan | None:
        if self._guard is None or not self._guard.is_available:
            return None

        capabilities = catalog.describe_for_planner()
        parts = [f"Available capabilities:\n{capabilities}"]
        if memory_context:
            parts.append(f"\nContext you already know:\n{memory_context}")
        parts.append(f"\nGoal: {goal}\n\nJSON:")
        prompt = "\n".join(parts)

        try:
            response = await self._guard.generate(
                prompt, system_prompt=_PLANNER_SYSTEM_PROMPT,
            )
            raw = response.content or ""
        except Exception:  # noqa: BLE001 - guard shouldn't raise, belt-and-braces
            _logger.debug("Planner LLM call failed", exc_info=True)
            return None

        data = self._extract_json_object(raw)
        if data is None:
            _logger.debug("Planner LLM returned no parseable JSON; using heuristic")
            return None

        return self._plan_from_json(goal, data, catalog)

    def _plan_from_json(
        self, goal: str, data: dict[str, Any], catalog: ICapabilityCatalog,
    ) -> Plan | None:
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return None

        nodes: list[TaskNode] = []
        seen_ids: set[str] = set()
        known = set(catalog.names())

        for i, raw in enumerate(raw_steps[: self._max_steps]):
            if not isinstance(raw, dict):
                continue
            node_id = self._clean_id(raw.get("id"), fallback=f"s{i + 1}", seen=seen_ids)
            description = self._clean_str(raw.get("description")) or goal
            tool = self._clean_str(raw.get("tool")).lower() or "reasoning"
            # Snap an unknown/hallucinated capability to reasoning so the
            # invoker always has a valid backend to resolve.
            if tool not in known:
                tool = "reasoning"
            deps = self._clean_deps(raw.get("depends_on"), seen_ids)
            confidence = self._clean_float(raw.get("confidence"), default=0.7)
            cost = self._clean_float(raw.get("cost"), default=1.0)

            nodes.append(
                TaskNode(
                    id=node_id,
                    description=description,
                    dependencies=deps,
                    required_tool=tool,
                    confidence=confidence,
                    estimated_cost=cost,
                    priority=len(raw_steps) - i,  # earlier steps score higher
                    retry_policy=self._default_retry,
                )
            )
            seen_ids.add(node_id)

        if not nodes:
            return None

        overall = self._clean_float(data.get("confidence"), default=0.0)
        if overall <= 0.0:
            overall = sum(n.confidence for n in nodes) / len(nodes)

        return Plan(
            goal=goal,
            nodes=tuple(nodes),
            overall_confidence=max(0.0, min(1.0, overall)),
            strategy="llm",
        )

    # ------------------------------------------------------------------
    # Heuristic path (deterministic, no LLM)
    # ------------------------------------------------------------------

    def _plan_heuristic(self, goal: str, catalog: ICapabilityCatalog) -> Plan:
        """Split a compound goal into sequential steps; else a single step.

        Each segment is mapped to a capability by keyword; unknown/unavailable
        capabilities fall back to ``reasoning`` (the always-safe floor).  Steps
        run sequentially (each depends on the previous) to preserve narrative
        order for compound requests like "search X and summarise it".
        """
        segments = [s.strip() for s in _STEP_SPLITTER.split(goal) if s.strip()]
        if len(segments) <= 1:
            tool = self._infer_capability(goal, catalog)
            node = TaskNode(
                id="s1", description=goal, required_tool=tool,
                confidence=0.6, estimated_cost=1.0,
                retry_policy=self._default_retry,
            )
            return Plan(goal=goal, nodes=(node,), overall_confidence=0.6,
                        strategy="single")

        nodes: list[TaskNode] = []
        prev_id: str | None = None
        for i, segment in enumerate(segments[: self._max_steps], start=1):
            node_id = f"s{i}"
            tool = self._infer_capability(segment, catalog)
            deps = [prev_id] if prev_id else []
            nodes.append(
                TaskNode(
                    id=node_id, description=segment, dependencies=deps,
                    required_tool=tool, confidence=0.55, estimated_cost=1.0,
                    priority=len(segments) - i, retry_policy=self._default_retry,
                )
            )
            prev_id = node_id

        return Plan(goal=goal, nodes=tuple(nodes), overall_confidence=0.55,
                    strategy="heuristic")

    @staticmethod
    def _infer_capability(text: str, catalog: ICapabilityCatalog) -> str:
        """Keyword-map *text* to an AVAILABLE capability, else 'reasoning'."""
        low = text.lower()
        # Ordered most-specific → least; only pick if actually available.
        candidates: list[tuple[tuple[str, ...], str]] = [
            (("calculate", "compute", "evaluate", "solve", "+", "*", "how much is"), "calculator"),
            (("search", "news", "latest", "weather", "current", "look up", "google"), "internet"),
            (("remember", "recall", "what did", "my ", "i like", "preference"), "memory"),
            (("file", "directory", "folder", "read the", "list the"), "filesystem"),
            (("open ", "http://", "https://", "website", "web page"), "browser"),
            (("time", "date", "today"), "datetime"),
        ]
        for keywords, cap in candidates:
            if any(k in low for k in keywords) and catalog.is_available(cap):
                return cap
        return "reasoning" if catalog.is_available("reasoning") else "reasoning"

    # ------------------------------------------------------------------
    # Parsing helpers (mirror ReflectionEngine's defensive style)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any] | None:
        raw = (raw or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw[raw.find("{"):] if "{" in raw else raw
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                obj = json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                return None
        return obj if isinstance(obj, dict) else None

    @staticmethod
    def _clean_str(value: Any) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _clean_id(value: Any, *, fallback: str, seen: set[str]) -> str:
        raw = value.strip() if isinstance(value, str) else ""
        candidate = re.sub(r"[^A-Za-z0-9_]", "_", raw)[:32] or fallback
        # Ensure uniqueness.
        if candidate in seen:
            n = 2
            while f"{candidate}_{n}" in seen:
                n += 1
            candidate = f"{candidate}_{n}"
        return candidate

    @staticmethod
    def _clean_deps(value: Any, seen: set[str]) -> list[str]:
        if not isinstance(value, list):
            return []
        deps: list[str] = []
        for item in value:
            if isinstance(item, str):
                dep = re.sub(r"[^A-Za-z0-9_]", "_", item.strip())[:32]
                # Only keep dependencies on already-declared, earlier steps —
                # this structurally prevents forward refs and cycles.
                if dep and dep in seen and dep not in deps:
                    deps.append(dep)
        return deps

    @staticmethod
    def _clean_float(value: Any, *, default: float) -> float:
        try:
            if isinstance(value, bool):  # bool is an int subclass — reject
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
