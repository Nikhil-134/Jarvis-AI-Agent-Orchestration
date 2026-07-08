"""Domain models for the Planning & Task Execution subsystem.

These are pure data types — no behaviour, no I/O, no dependencies on the
runtime, orchestrator, tools, or memory layers.  They form the shared
vocabulary that the planner, task graph, executor, verifier, and coordinator
all speak.

Design notes
------------
* :class:`TaskNode` is **mutable** — its ``status`` and ``result`` change as the
  executor runs it.  Identity is defined by ``id`` so nodes can be used as dict
  keys / set members during scheduling.
* :class:`RetryPolicy`, :class:`Plan`, :class:`TaskMetrics`,
  :class:`ExecutionMetrics`, :class:`NodeResult`, :class:`VerificationResult`,
  and :class:`PlanningOutcome` are **frozen** — once produced they are read-only
  snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Lifecycle state of a single task node.

    ::

        PENDING ──► READY ──► RUNNING ──► SUCCEEDED
           │                     │  │
           │                     │  ├─► FAILED
           │                     │  └─► TIMED_OUT
           └────────────────────►└────► SKIPPED / CANCELLED
    """

    PENDING = "pending"      # created, dependencies not yet satisfied
    READY = "ready"          # dependencies satisfied, awaiting a worker
    RUNNING = "running"      # currently executing
    SUCCEEDED = "succeeded"  # completed successfully
    FAILED = "failed"        # executed but the backend reported failure
    SKIPPED = "skipped"      # a dependency failed, so this can never run
    CANCELLED = "cancelled"  # cancelled before/while running
    TIMED_OUT = "timed_out"  # exceeded its per-task timeout on every attempt

    @property
    def is_terminal(self) -> bool:
        """Whether no further transition is possible from this state."""
        return self in _TERMINAL_STATES

    @property
    def is_success(self) -> bool:
        return self is TaskStatus.SUCCEEDED


_TERMINAL_STATES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.SUCCEEDED,
        TaskStatus.FAILED,
        TaskStatus.SKIPPED,
        TaskStatus.CANCELLED,
        TaskStatus.TIMED_OUT,
    }
)

_FAILURE_STATES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED}
)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How a task should be retried on transient failure/timeout.

    ``backoff_seconds * multiplier ** attempt`` is waited before each retry.
    ``max_retries == 0`` means a single attempt with no retries.
    """

    max_retries: int = 1
    backoff_seconds: float = 0.5
    multiplier: float = 2.0

    def delay_for_attempt(self, attempt: int) -> float:
        """Return the backoff delay (seconds) before *attempt* (0-indexed)."""
        if attempt < 0:
            return 0.0
        return self.backoff_seconds * (self.multiplier ** attempt)

    @classmethod
    def none(cls) -> "RetryPolicy":
        """A policy that performs exactly one attempt (no retries)."""
        return cls(max_retries=0, backoff_seconds=0.0, multiplier=1.0)


@dataclass(frozen=True, slots=True)
class NodeResult:
    """The terminal outcome of executing one :class:`TaskNode`.

    Frozen snapshot attached to the node after execution and stored in the
    :class:`~planning.scratchpad.ReasoningScratchpad`.
    """

    node_id: str
    status: TaskStatus
    output: str = ""
    error: str | None = None
    backend: str = ""
    attempts: int = 0
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.status is TaskStatus.SUCCEEDED


class TaskNode:
    """A single unit of work in a :class:`~planning.task_graph.TaskGraph`.

    Mutable: the executor updates :attr:`status` and attaches :attr:`result`.
    Equality/hash are by :attr:`id` so a node can key scheduling dictionaries.
    """

    __slots__ = (
        "id",
        "description",
        "dependencies",
        "priority",
        "status",
        "retry_policy",
        "estimated_cost",
        "required_tool",
        "confidence",
        "args",
        "critical",
        "result",
    )

    def __init__(
        self,
        id: str,
        description: str,
        *,
        dependencies: list[str] | None = None,
        priority: int = 0,
        retry_policy: RetryPolicy | None = None,
        estimated_cost: float = 1.0,
        required_tool: str = "reasoning",
        confidence: float = 1.0,
        args: dict[str, Any] | None = None,
        critical: bool = True,
        status: TaskStatus = TaskStatus.PENDING,
    ) -> None:
        self.id = id
        self.description = description
        self.dependencies: list[str] = list(dependencies or [])
        self.priority = priority
        self.status = status
        self.retry_policy = retry_policy or RetryPolicy()
        self.estimated_cost = estimated_cost
        self.required_tool = required_tool
        self.confidence = confidence
        self.args: dict[str, Any] = dict(args or {})
        # A critical node's failure skips its dependents; a non-critical node's
        # failure is tolerated (dependents may still run).
        self.critical = critical
        self.result: NodeResult | None = None

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TaskNode) and other.id == self.id

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"TaskNode(id={self.id!r}, tool={self.required_tool!r}, "
            f"status={self.status.value}, deps={self.dependencies})"
        )


@dataclass(frozen=True, slots=True)
class Plan:
    """A decomposition of a goal into an ordered set of task nodes."""

    goal: str
    nodes: tuple[TaskNode, ...]
    overall_confidence: float = 1.0
    strategy: str = "llm"  # "llm" | "heuristic" | "single"

    @property
    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    @property
    def is_multi_step(self) -> bool:
        return len(self.nodes) > 1


@dataclass(frozen=True, slots=True)
class TaskMetrics:
    """Per-task execution metrics."""

    node_id: str
    required_tool: str
    status: TaskStatus
    attempts: int
    duration_ms: float


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    """Aggregate metrics for a full :class:`Plan` execution."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    cancelled: int = 0
    timed_out: int = 0
    total_attempts: int = 0
    wall_time_ms: float = 0.0
    per_task: tuple[TaskMetrics, ...] = ()

    @property
    def all_succeeded(self) -> bool:
        return self.total > 0 and self.succeeded == self.total

    @property
    def success_ratio(self) -> float:
        return (self.succeeded / self.total) if self.total else 0.0

    def summary(self) -> str:
        """A compact one-line summary for logs."""
        return (
            f"{self.succeeded}/{self.total} ok"
            f" (failed={self.failed}, skipped={self.skipped}, "
            f"cancelled={self.cancelled}, timed_out={self.timed_out}, "
            f"attempts={self.total_attempts}, {self.wall_time_ms:.0f}ms)"
        )


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of :class:`~planning.verifier.ResponseVerifier` validation."""

    ok: bool
    response: str
    confidence: float
    issues: tuple[str, ...] = ()

    @property
    def had_issues(self) -> bool:
        return len(self.issues) > 0


@dataclass(frozen=True, slots=True)
class PlanningOutcome:
    """The result of :meth:`~planning.coordinator.PlanningCoordinator.run`.

    ``accepted`` tells the runtime whether the planning path produced a usable
    answer.  When ``False`` the runtime falls through to its regex routing
    fallback — the planning subsystem never dead-ends a request.
    """

    accepted: bool
    response: str
    plan: Plan | None = None
    metrics: ExecutionMetrics | None = None
    verification: VerificationResult | None = None
    reason: str = ""
