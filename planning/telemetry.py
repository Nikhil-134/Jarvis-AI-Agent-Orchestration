"""Structured telemetry for the Planning & Task Execution subsystem.

The runtime already *logs* (plain ``_logger.info`` lines) and already exposes a
live *progress callback* (``(event, node_id, detail)`` string tuples for a UI).
Telemetry is a third, distinct concern: **machine-readable observability
records** for planner decisions, tool execution, retries, latency, and fallback
reasons (the five signals the subsystem is required to surface).

Design (SOLID / DI)
-------------------
* :class:`ITelemetrySink` is the Dependency-Inversion seam — the emitter depends
  on this abstraction, never a concrete backend.
* :class:`NullTelemetrySink` is the default: a true no-op, so telemetry adds
  **zero** overhead and cannot change behaviour when disabled.
* :class:`LoggingTelemetrySink` serialises each event to one JSON line on a
  dedicated ``planning.telemetry`` logger, which flows into the project's
  existing rotating **local** log file — ₹0, no new infrastructure, no network.
* :class:`InMemoryTelemetrySink` retains events for tests and in-process metrics
  export.
* :class:`PlanningTelemetry` is the typed façade the executor and coordinator
  call.  Every method is guaranteed **never to raise** — a telemetry failure
  must never break planning — and is a cheap no-op when the sink is ``None``.

Why not the orchestrator ``MessageBus``?  That bus carries ``AgentMessage``
objects between agents; telemetry is not an agent message, and coupling the
executor's hot loop to an async pub/sub would add latency and an unrelated
dependency.  A synchronous sink keeps emission free on the fast path.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_logger = logging.getLogger(__name__)

# Dedicated telemetry logger — routes into the same rotating file handler the
# root logger already configures (see config/logging_config.py).  Kept separate
# so a deployment can raise/lower telemetry verbosity independently.
_TELEMETRY_LOGGER_NAME = "planning.telemetry"


class TelemetryKind(str, Enum):
    """The kinds of structured event the planning subsystem emits.

    Mapped to the five required observability signals:

    ==================  ==================================================
    signal              event kind(s)
    ==================  ==================================================
    planner decisions   ``PLAN_DECIDED``
    tool execution      ``TASK_STARTED`` + ``TASK_COMPLETED``
    retries             ``TASK_RETRY``
    latency             ``duration_ms`` on ``TASK_COMPLETED``;
                        ``wall_time_ms`` on ``RUN_COMPLETED``
    fallback reasons    ``PLAN_DECLINED`` (pre-exec) + ``RUN_COMPLETED``
                        (``accepted=false`` with a ``reason``)
    ==================  ==================================================
    """

    PLAN_DECIDED = "plan_decided"
    PLAN_DECLINED = "plan_declined"
    TASK_STARTED = "task_started"
    TASK_RETRY = "task_retry"
    TASK_COMPLETED = "task_completed"
    RUN_COMPLETED = "run_completed"


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    """One immutable, machine-readable telemetry record.

    :param kind: the :class:`TelemetryKind` value (a stable string).
    :param timestamp: wall-clock seconds since the epoch when the event fired.
    :param fields: flat mapping of primitive attributes (str/int/float/bool/None)
        describing the event; kept flat so it serialises to a single JSON object.
    """

    kind: str
    timestamp: float
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-ready dict of the whole event."""
        payload: dict[str, Any] = {"event": self.kind, "ts": round(self.timestamp, 3)}
        payload.update(self.fields)
        return payload

    def to_json(self) -> str:
        """Serialise the event to one compact JSON line.

        Non-serialisable values are coerced with ``str`` so emission can never
        raise on an unexpected field type.
        """
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


class ITelemetrySink(ABC):
    """A destination for :class:`TelemetryEvent` records (DIP seam)."""

    @abstractmethod
    def emit(self, event: TelemetryEvent) -> None:
        """Record *event*. Implementations must not raise."""


class NullTelemetrySink(ITelemetrySink):
    """The default sink: discards every event (zero overhead)."""

    def emit(self, event: TelemetryEvent) -> None:  # noqa: D102 - trivial
        return None


class LoggingTelemetrySink(ITelemetrySink):
    """Writes each event as one JSON line to the local ``planning.telemetry`` log.

    Reuses the project's existing rotating-file logging — no network, no new
    dependency, ₹0.  Defaults to ``INFO`` so telemetry lands in the file handler
    (which captures ``INFO`` and above) but stays off the ``WARNING``-only
    console.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        level: int = logging.INFO,
    ) -> None:
        self._logger = logger or logging.getLogger(_TELEMETRY_LOGGER_NAME)
        self._level = level

    def emit(self, event: TelemetryEvent) -> None:
        try:
            self._logger.log(self._level, "%s", event.to_json())
        except Exception:  # noqa: BLE001 - telemetry must never break the caller
            # Deliberately swallowed: a logging failure cannot be allowed to
            # propagate into the planner/executor.  Reported once at debug.
            _logger.debug("Telemetry emission failed", exc_info=True)


class InMemoryTelemetrySink(ITelemetrySink):
    """Collects events in memory — for tests and in-process metrics export."""

    def __init__(self, *, max_events: int = 10_000) -> None:
        self._events: list[TelemetryEvent] = []
        self._max = max_events

    def emit(self, event: TelemetryEvent) -> None:
        self._events.append(event)
        # Bound memory: drop the oldest if we somehow exceed the cap.
        if len(self._events) > self._max:
            del self._events[: len(self._events) - self._max]

    @property
    def events(self) -> tuple[TelemetryEvent, ...]:
        """All recorded events, in emission order."""
        return tuple(self._events)

    def of_kind(self, kind: TelemetryKind | str) -> tuple[TelemetryEvent, ...]:
        """Return only the events whose ``kind`` matches *kind*."""
        wanted = kind.value if isinstance(kind, TelemetryKind) else kind
        return tuple(e for e in self._events if e.kind == wanted)

    def clear(self) -> None:
        self._events.clear()


class PlanningTelemetry:
    """Typed, non-raising façade the executor and coordinator emit through.

    Wraps an optional :class:`ITelemetrySink`.  When constructed with ``None``
    (the default) it uses a :class:`NullTelemetrySink`, so every call is a cheap
    no-op and the subsystem behaves exactly as if telemetry did not exist.

    An injectable *clock* keeps timestamps deterministic under test.
    """

    def __init__(
        self,
        sink: ITelemetrySink | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._sink: ITelemetrySink = sink or NullTelemetrySink()
        self._clock = clock or time.time
        self._enabled = not isinstance(self._sink, NullTelemetrySink)

    @property
    def enabled(self) -> bool:
        """Whether a real (non-null) sink is attached."""
        return self._enabled

    # ------------------------------------------------------------------
    # Planner-decision + run-level signals (emitted by the coordinator)
    # ------------------------------------------------------------------

    def plan_decided(
        self, *, goal: str, strategy: str, node_count: int, confidence: float,
    ) -> None:
        """Record that the planner produced a plan the coordinator accepted."""
        self._emit(
            TelemetryKind.PLAN_DECIDED,
            goal=_clip(goal),
            strategy=strategy,
            node_count=node_count,
            confidence=round(confidence, 4),
        )

    def plan_declined(self, *, goal: str, reason: str, confidence: float) -> None:
        """Record that planning declined the goal (→ regex fallback)."""
        self._emit(
            TelemetryKind.PLAN_DECLINED,
            goal=_clip(goal),
            reason=reason,
            confidence=round(confidence, 4),
        )

    def run_completed(
        self,
        *,
        goal: str,
        accepted: bool,
        reason: str,
        wall_time_ms: float,
        succeeded: int,
        total: int,
        verification_confidence: float | None = None,
    ) -> None:
        """Record the terminal outcome of a full planning run (incl. latency)."""
        self._emit(
            TelemetryKind.RUN_COMPLETED,
            goal=_clip(goal),
            accepted=accepted,
            reason=reason,
            wall_time_ms=round(wall_time_ms, 2),
            succeeded=succeeded,
            total=total,
            verification_confidence=(
                None if verification_confidence is None
                else round(verification_confidence, 4)
            ),
        )

    # ------------------------------------------------------------------
    # Per-task signals (emitted by the executor)
    # ------------------------------------------------------------------

    def task_started(self, *, node_id: str, tool: str, description: str) -> None:
        """Record that a task began executing against its backend/tool."""
        self._emit(
            TelemetryKind.TASK_STARTED,
            node_id=node_id,
            tool=tool,
            description=_clip(description, 120),
        )

    def task_retry(
        self, *, node_id: str, tool: str, attempt: int, reason: str,
    ) -> None:
        """Record a retry decision for a task (1-indexed *attempt* just tried)."""
        self._emit(
            TelemetryKind.TASK_RETRY,
            node_id=node_id,
            tool=tool,
            attempt=attempt,
            reason=_clip(reason, 160),
        )

    def task_completed(
        self,
        *,
        node_id: str,
        tool: str,
        status: str,
        attempts: int,
        duration_ms: float,
        error: str | None = None,
    ) -> None:
        """Record a task's terminal outcome, backend, attempts, and latency."""
        self._emit(
            TelemetryKind.TASK_COMPLETED,
            node_id=node_id,
            tool=tool,
            status=status,
            attempts=attempts,
            duration_ms=round(duration_ms, 2),
            error=_clip(error, 160) if error else None,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _emit(self, kind: TelemetryKind, **fields: Any) -> None:
        """Build and dispatch an event. Never raises."""
        if not self._enabled:
            return
        try:
            event = TelemetryEvent(
                kind=kind.value, timestamp=self._clock(), fields=fields,
            )
            self._sink.emit(event)
        except Exception:  # noqa: BLE001 - telemetry must never break the caller
            _logger.debug("Telemetry build/emit failed for %s", kind, exc_info=True)


def _clip(text: str, limit: int = 200) -> str:
    """Trim *text* for telemetry so a huge goal/description can't bloat a log line."""
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_telemetry_sink(
    enabled: bool, *, logger: logging.Logger | None = None,
) -> ITelemetrySink:
    """Return a :class:`LoggingTelemetrySink` when *enabled*, else a null sink.

    The factory the DI composition root uses so callers never branch on the flag
    themselves.
    """
    return LoggingTelemetrySink(logger) if enabled else NullTelemetrySink()
