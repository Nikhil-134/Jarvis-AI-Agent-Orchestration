"""CapabilityCatalog — the confidence-based routing table.

This is the heart of requirement #4 ("confidence-based intent routing — the
planner decides which tool(s) to invoke").  It answers two questions:

1. **For the planner:** what capabilities exist right now, and what does each
   do?  (:meth:`describe_for_planner`)
2. **For the invoker:** given a capability name a task asked for, which concrete
   backend runs it, is it available, and does it need permission?
   (:meth:`resolve`)

Availability is computed from **real** state, not an idealised list:
* ``memory`` / ``internet`` / ``reasoning`` are available iff their injected
  services are present (and, for internet, ``.available``).
* tool-backed capabilities (``calculator``, ``filesystem``, ``browser``,
  ``desktop``, ``plugins``) are available iff the concrete tool is actually
  registered in the :class:`ToolRegistry` (some need absent native deps).
* ``python`` has **no** backend (there is no python tool, and silently routing
  it to the DANGEROUS ``shell`` tool would be unsafe) → always unavailable.

DANGEROUS-permissioned tools are flagged ``requires_permission`` so the invoker
can refuse them unless auto-approve is on (avoids a stdin hang / permission
raise inside the async pipeline).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tools.interfaces import PermissionLevel

if TYPE_CHECKING:
    from knowledge.internet import InternetKnowledgeService
    from memory import MemoryService
    from tools.engine import ToolExecutionEngine

_logger = logging.getLogger(__name__)

# Logical backends a capability can resolve to.
BACKEND_MEMORY = "memory"
BACKEND_INTERNET = "internet"
BACKEND_REASONING = "reasoning"
BACKEND_TOOL = "tool"
BACKEND_NONE = "none"


@dataclass(frozen=True, slots=True)
class ResolvedCapability:
    """How a capability name maps to something the invoker can run."""

    name: str
    backend: str                 # one of BACKEND_*
    description: str
    available: bool
    tool_name: str = ""          # concrete tool for BACKEND_TOOL
    requires_permission: bool = False  # DANGEROUS tool → needs auto-approve
    default_arg: str = ""        # payload key the goal text maps to (e.g. "expression")
    reason: str = ""             # why unavailable (internal log only)


# Static capability vocabulary.  ``tool_name`` links to a concrete registered
# tool; the empty entries are service-backed.  ``python`` is intentionally
# backendless.  Keep the names aligned with the planner prompt + the runtime
# intent labels (memory/internet/calculator/python/browser/desktop/filesystem).
_CAPABILITY_TABLE: dict[str, dict] = {
    "memory": dict(
        backend=BACKEND_MEMORY,
        description="Recall relevant facts, preferences and past conversation from local memory.",
    ),
    "internet": dict(
        backend=BACKEND_INTERNET,
        description="Fetch fresh public facts (weather/news/current events) — only when required.",
    ),
    "reasoning": dict(
        backend=BACKEND_REASONING,
        description="Reason, explain, summarise or write prose using the local language model.",
    ),
    "calculator": dict(
        backend=BACKEND_TOOL, tool_name="calculator", default_arg="expression",
        description="Evaluate a arithmetic/math expression safely.",
    ),
    "filesystem": dict(
        backend=BACKEND_TOOL, tool_name="file_system", default_arg="path",
        description="Read/inspect the local file system.",
    ),
    "browser": dict(
        backend=BACKEND_TOOL, tool_name="browser", default_arg="url",
        description="Open or fetch a web page.",
    ),
    "desktop": dict(
        backend=BACKEND_TOOL, tool_name="system_info", default_arg="",
        description="Inspect the local machine (system info).",
    ),
    "datetime": dict(
        backend=BACKEND_TOOL, tool_name="datetime", default_arg="",
        description="Get the current date and time.",
    ),
    "hash": dict(
        backend=BACKEND_TOOL, tool_name="hash", default_arg="data",
        description="Compute a cryptographic hash of some text.",
    ),
    "python": dict(
        backend=BACKEND_NONE,
        description="(unavailable) Execute arbitrary Python — no safe local backend exists.",
    ),
}


class CapabilityCatalog:
    """Real-time catalogue of executable capabilities (SOLID: one job).

    Constructed once per :meth:`~planning.coordinator.PlanningCoordinator.run`
    from the injected subsystems so availability always reflects the live
    system.
    """

    def __init__(
        self,
        *,
        tool_engine: "ToolExecutionEngine | None" = None,
        memory_service: "MemoryService | None" = None,
        internet_service: "InternetKnowledgeService | None" = None,
        reasoning_available: bool = False,
        allow_dangerous: bool = False,
    ) -> None:
        self._tool_engine = tool_engine
        self._memory = memory_service
        self._internet = internet_service
        self._reasoning_available = reasoning_available
        self._allow_dangerous = allow_dangerous
        self._resolved: dict[str, ResolvedCapability] = {}
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _registered_tool_names(self) -> set[str]:
        if self._tool_engine is None:
            return set()
        try:
            return {spec.name for spec in self._tool_engine.registry.list_specs()}
        except Exception:  # noqa: BLE001 - defensive; never break planning
            _logger.debug("Failed to list tool specs for catalog", exc_info=True)
            return set()

    def _tool_is_dangerous(self, tool_name: str) -> bool:
        if self._tool_engine is None:
            return False
        try:
            tool = self._tool_engine.registry.get(tool_name)
            return tool is not None and tool.permission_level == PermissionLevel.DANGEROUS
        except Exception:  # noqa: BLE001
            return False

    def _build(self) -> None:
        registered = self._registered_tool_names()

        for name, spec in _CAPABILITY_TABLE.items():
            backend = spec["backend"]
            description = spec["description"]
            tool_name = spec.get("tool_name", "")
            default_arg = spec.get("default_arg", "")

            available = False
            requires_permission = False
            reason = ""

            if backend == BACKEND_MEMORY:
                available = self._memory is not None
                reason = "" if available else "no memory service"
            elif backend == BACKEND_INTERNET:
                available = self._internet is not None and bool(
                    getattr(self._internet, "available", False)
                )
                reason = "" if available else "internet service unavailable"
            elif backend == BACKEND_REASONING:
                available = self._reasoning_available
                reason = "" if available else "no language model"
            elif backend == BACKEND_TOOL:
                if tool_name in registered:
                    dangerous = self._tool_is_dangerous(tool_name)
                    requires_permission = dangerous and not self._allow_dangerous
                    # A dangerous tool without auto-approve is treated as
                    # unavailable to the autonomous planner (honest, no hang).
                    available = not requires_permission
                    if not available:
                        reason = "requires permission (auto-approve off)"
                else:
                    available = False
                    reason = f"tool {tool_name!r} not registered"
            else:  # BACKEND_NONE (python)
                available = False
                reason = "no safe local backend"

            self._resolved[name] = ResolvedCapability(
                name=name,
                backend=backend,
                description=description,
                available=available,
                tool_name=tool_name,
                requires_permission=requires_permission,
                default_arg=default_arg,
                reason=reason,
            )

    # ------------------------------------------------------------------
    # Public API (ICapabilityCatalog)
    # ------------------------------------------------------------------

    def resolve(self, capability: str) -> ResolvedCapability:
        """Return how *capability* maps to a backend.

        Unknown capabilities resolve to an unavailable ``reasoning`` fallback so
        the invoker degrades gracefully rather than raising.
        """
        cap = (capability or "").strip().lower()
        if cap in self._resolved:
            return self._resolved[cap]
        # Unknown capability: default to reasoning if the LM is up, else none.
        if self._reasoning_available:
            return ResolvedCapability(
                name=cap or "reasoning",
                backend=BACKEND_REASONING,
                description="Reason about the task using the local model.",
                available=True,
                reason="unknown capability → reasoning fallback",
            )
        return ResolvedCapability(
            name=cap or "unknown",
            backend=BACKEND_NONE,
            description="",
            available=False,
            reason="unknown capability, no reasoning backend",
        )

    def is_available(self, capability: str) -> bool:
        return self.resolve(capability).available

    def names(self) -> tuple[str, ...]:
        return tuple(self._resolved.keys())

    def available_names(self) -> tuple[str, ...]:
        return tuple(n for n, c in self._resolved.items() if c.available)

    def describe_for_planner(self) -> str:
        """Return a compact, prompt-ready description of AVAILABLE capabilities."""
        lines: list[str] = []
        for name, cap in self._resolved.items():
            if cap.available:
                lines.append(f"- {name}: {cap.description}")
        if not lines:
            # Reasoning is the always-safe floor; if nothing else, say so.
            return "- reasoning: Answer using the local language model."
        return "\n".join(lines)
