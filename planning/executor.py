"""TaskExecutor — concurrent, resilient execution of a plan's task graph.

Requirement #3: parallel execution, dependency resolution, retries,
cancellation, timeout handling, and progress tracking.

Scheduling model — **continuous scheduling with a semaphore** (deliberately
different from :class:`orchestrator.workflow.WorkflowEngine`, which is
wave-based and blocks on the slowest node in each wave).  Here:

* a node becomes eligible the instant *all* its dependencies complete;
* up to ``max_parallel`` nodes run at once (``asyncio.Semaphore``);
* the scheduler waits on ``asyncio.wait(..., FIRST_COMPLETED)`` and immediately
  schedules any newly-ready nodes — no head-of-line blocking between a fast
  local tool and a slow model call.

Resilience
----------
* **Per-task timeout** via ``asyncio.wait_for`` (backstop; the ToolInvoker also
  pushes the timeout into the tool engine so it is authoritative there).
* **Retries** per the node's :class:`RetryPolicy` with exponential backoff.
* **Cancellation** via a shared ``asyncio.Event`` — new nodes stop scheduling
  and in-flight tasks are cancelled cooperatively.
* **Failure isolation** — a *critical* node's failure skips its transitive
  dependents (:meth:`TaskGraph.skip_unreachable`); the run still terminates and
  returns metrics.
* **Never raises** — ``_run_node`` converts every outcome into a
  :class:`NodeResult`; the scheduler always returns :class:`ExecutionMetrics`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from planning.interfaces import ITaskExecutor, IToolInvoker
from planning.models import (
    ExecutionMetrics,
    NodeResult,
    Plan,
    TaskMetrics,
    TaskNode,
    TaskStatus,
)
from planning.task_graph import TaskGraph
from planning.telemetry import PlanningTelemetry

if TYPE_CHECKING:
    from planning.scratchpad import ReasoningScratchpad

_logger = logging.getLogger(__name__)

# Callback signature: (event, node_id, detail) for progress reporting.
ProgressCallback = Callable[[str, str, str], None]


class TaskExecutor(ITaskExecutor):
    """Executes a :class:`Plan` concurrently with retries, timeouts, cancel."""

    def __init__(
        self,
        invoker: IToolInvoker,
        *,
        max_parallel: int = 4,
        task_timeout_seconds: float = 30.0,
        scratchpad: "ReasoningScratchpad | None" = None,
        progress: ProgressCallback | None = None,
        telemetry: PlanningTelemetry | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._invoker = invoker
        self._max_parallel = max(1, max_parallel)
        self._timeout = task_timeout_seconds
        self._scratchpad = scratchpad
        self._progress = progress
        # Telemetry is an observability side-channel: a null facade by default
        # so behaviour and latency are unchanged when telemetry is disabled.
        self._telemetry = telemetry or PlanningTelemetry()
        self._clock = clock or time.monotonic
        self._cancel = asyncio.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cooperative cancellation of an in-progress execution."""
        self._cancel.set()

    async def execute(self, plan: Plan) -> ExecutionMetrics:
        """Execute *plan* and return aggregate :class:`ExecutionMetrics`."""
        self._cancel = asyncio.Event()
        started = self._clock()

        graph = TaskGraph(list(plan.nodes))
        if graph.is_empty:
            return ExecutionMetrics(wall_time_ms=0.0)

        # Validate once; a malformed graph is reported as all-failed metrics
        # rather than raising into the coordinator.
        try:
            graph.validate()
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Plan graph invalid, aborting execution: %s", exc)
            return self._invalid_metrics(graph, started)

        completed_ids: set[str] = set()
        sem = asyncio.Semaphore(self._max_parallel)
        in_flight: dict[asyncio.Task, TaskNode] = {}

        while True:
            # Cancellation → stop scheduling; cancel any in-flight tasks.
            if self._cancel.is_set():
                await self._drain_cancelled(in_flight, graph, completed_ids)
                break

            # Schedule every newly-ready node.
            for node in graph.ready_nodes(completed_ids):
                node.status = TaskStatus.READY
                self._emit("scheduled", node.id, node.required_tool)
                task = asyncio.ensure_future(self._run_node(node, sem))
                in_flight[task] = node

            if not in_flight:
                # Nothing running and nothing ready. Either done, or the
                # remaining nodes are blocked by failures → mark them skipped
                # (deadlock guard so the loop always terminates).
                if graph.pending_nodes:
                    for node in graph.pending_nodes:
                        node.status = TaskStatus.SKIPPED
                        self._emit("skipped", node.id, "unreachable")
                break

            done, _pending = await asyncio.wait(
                in_flight.keys(), return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                node = in_flight.pop(task)
                result = task.result()  # _run_node never raises
                self._finish_node(graph, node, result, completed_ids)

        return self._collect_metrics(graph, started)

    # ------------------------------------------------------------------
    # Node execution
    # ------------------------------------------------------------------

    async def _run_node(self, node: TaskNode, sem: asyncio.Semaphore) -> NodeResult:
        """Run one node with retries + timeout. Never raises."""
        async with sem:
            node.status = TaskStatus.RUNNING
            self._emit("running", node.id, node.description[:60])
            self._telemetry.task_started(
                node_id=node.id, tool=node.required_tool, description=node.description,
            )
            policy = node.retry_policy
            attempts = 0
            last_error: str | None = None
            start = self._clock()
            context = self._dependency_context(node)

            for attempt in range(policy.max_retries + 1):
                if self._cancel.is_set():
                    return self._terminal(node, TaskStatus.CANCELLED, start, attempts,
                                          error="cancelled")
                attempts += 1
                try:
                    result = await asyncio.wait_for(
                        self._invoker.invoke(node, context=context),
                        timeout=self._timeout,
                    )
                except asyncio.CancelledError:
                    # Cooperative cancel (e.g. from _drain_cancelled) — honour it.
                    return self._terminal(node, TaskStatus.CANCELLED, start, attempts,
                                          error="cancelled")
                except asyncio.TimeoutError:
                    last_error = f"timed out after {self._timeout}s"
                    _logger.warning("Node %s timed out (attempt %d/%d)",
                                    node.id, attempt + 1, policy.max_retries + 1)
                    if attempt < policy.max_retries and not self._cancel.is_set():
                        self._telemetry.task_retry(
                            node_id=node.id, tool=node.required_tool,
                            attempt=attempts, reason=last_error,
                        )
                        await asyncio.sleep(policy.delay_for_attempt(attempt))
                        continue
                    return self._terminal(node, TaskStatus.TIMED_OUT, start, attempts,
                                          error=last_error)
                except Exception as exc:  # noqa: BLE001 - invoker shouldn't raise
                    last_error = str(exc)
                    _logger.exception("Node %s raised (attempt %d)", node.id, attempt + 1)
                    if attempt < policy.max_retries and not self._cancel.is_set():
                        self._telemetry.task_retry(
                            node_id=node.id, tool=node.required_tool,
                            attempt=attempts, reason=last_error,
                        )
                        await asyncio.sleep(policy.delay_for_attempt(attempt))
                        continue
                    return self._terminal(node, TaskStatus.FAILED, start, attempts,
                                          error=last_error)

                # Got a NodeResult from the invoker.
                if result.status is TaskStatus.SUCCEEDED:
                    return self._reattempt(result, attempts, start)
                # Backend reported failure — retry if policy allows.
                last_error = result.error
                if attempt < policy.max_retries and not self._cancel.is_set():
                    self._telemetry.task_retry(
                        node_id=node.id, tool=node.required_tool,
                        attempt=attempts, reason=last_error or "backend failure",
                    )
                    await asyncio.sleep(policy.delay_for_attempt(attempt))
                    continue
                return self._reattempt(result, attempts, start)

            # Unreachable, but keep the type checker + safety happy.
            return self._terminal(node, TaskStatus.FAILED, start, attempts,
                                  error=last_error or "unknown")

    def _finish_node(
        self, graph: TaskGraph, node: TaskNode, result: NodeResult,
        completed_ids: set[str],
    ) -> None:
        node.result = result
        node.status = result.status
        completed_ids.add(node.id)
        if self._scratchpad is not None:
            self._scratchpad.record_result(result)
        self._emit(result.status.value, node.id, result.error or "ok")
        self._telemetry.task_completed(
            node_id=node.id,
            tool=node.required_tool,
            status=result.status.value,
            attempts=result.attempts,
            duration_ms=result.duration_ms,
            error=result.error,
        )

        # A failed/timed-out/cancelled *critical* node skips its dependents.
        if node.critical and result.status in (
            TaskStatus.FAILED, TaskStatus.TIMED_OUT, TaskStatus.CANCELLED,
        ):
            for skipped_id in graph.skip_unreachable(node.id):
                completed_ids.add(skipped_id)  # so ready_nodes won't wait on them
                if self._scratchpad is not None:
                    skipped_node = graph.get(skipped_id)
                    if skipped_node is not None:
                        self._scratchpad.record_result(
                            NodeResult(node_id=skipped_id, status=TaskStatus.SKIPPED,
                                       error="dependency failed")
                        )

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    async def _drain_cancelled(
        self, in_flight: dict[asyncio.Task, TaskNode], graph: TaskGraph,
        completed_ids: set[str],
    ) -> None:
        for task in in_flight:
            task.cancel()
        if in_flight:
            await asyncio.gather(*in_flight.keys(), return_exceptions=True)
        for task, node in in_flight.items():
            if node.result is None:
                result = NodeResult(node_id=node.id, status=TaskStatus.CANCELLED,
                                    error="cancelled")
                self._finish_node(graph, node, result, completed_ids)
        in_flight.clear()
        # Any remaining pending nodes are cancelled too.
        for node in graph.pending_nodes:
            node.status = TaskStatus.CANCELLED
            if self._scratchpad is not None:
                self._scratchpad.record_result(
                    NodeResult(node_id=node.id, status=TaskStatus.CANCELLED,
                               error="cancelled")
                )

    # ------------------------------------------------------------------
    # Metrics + helpers
    # ------------------------------------------------------------------

    def _collect_metrics(self, graph: TaskGraph, started: float) -> ExecutionMetrics:
        per_task: list[TaskMetrics] = []
        counts = {s: 0 for s in TaskStatus}
        total_attempts = 0
        for node in graph.nodes:
            status = node.status
            counts[status] = counts.get(status, 0) + 1
            attempts = node.result.attempts if node.result else 0
            duration = node.result.duration_ms if node.result else 0.0
            total_attempts += attempts
            per_task.append(TaskMetrics(
                node_id=node.id, required_tool=node.required_tool,
                status=status, attempts=attempts, duration_ms=duration,
            ))
        wall = (self._clock() - started) * 1000.0
        metrics = ExecutionMetrics(
            total=len(graph),
            succeeded=counts.get(TaskStatus.SUCCEEDED, 0),
            failed=counts.get(TaskStatus.FAILED, 0),
            skipped=counts.get(TaskStatus.SKIPPED, 0),
            cancelled=counts.get(TaskStatus.CANCELLED, 0),
            timed_out=counts.get(TaskStatus.TIMED_OUT, 0),
            total_attempts=total_attempts,
            wall_time_ms=wall,
            per_task=tuple(per_task),
        )
        _logger.info("Plan execution complete: %s", metrics.summary())
        return metrics

    def _invalid_metrics(self, graph: TaskGraph, started: float) -> ExecutionMetrics:
        for node in graph.nodes:
            node.status = TaskStatus.FAILED
            node.result = NodeResult(node_id=node.id, status=TaskStatus.FAILED,
                                     error="invalid plan graph")
        return self._collect_metrics(graph, started)

    @staticmethod
    def _terminal(
        node: TaskNode, status: TaskStatus, start: float, attempts: int,
        *, error: str | None,
    ) -> NodeResult:
        return NodeResult(
            node_id=node.id, status=status, output="", error=error,
            backend=node.required_tool, attempts=attempts,
            duration_ms=(time.monotonic() - start) * 1000.0,
        )

    @staticmethod
    def _reattempt(result: NodeResult, attempts: int, start: float) -> NodeResult:
        """Return *result* with the real attempt count + duration stamped on."""
        return NodeResult(
            node_id=result.node_id, status=result.status, output=result.output,
            error=result.error, backend=result.backend, attempts=attempts,
            duration_ms=(time.monotonic() - start) * 1000.0,
        )

    def _dependency_context(self, node: TaskNode) -> str:
        """Assemble the successful outputs of *node*'s dependencies as context."""
        if self._scratchpad is None or not node.dependencies:
            return ""
        parts: list[str] = []
        for dep_id in node.dependencies:
            dep_result = self._scratchpad.result_for(dep_id)
            if dep_result is not None and dep_result.success and dep_result.output:
                parts.append(f"Result of previous step: {dep_result.output}")
        return "\n\n".join(parts)

    def _emit(self, event: str, node_id: str, detail: str) -> None:
        if self._progress is not None:
            try:
                self._progress(event, node_id, detail)
            except Exception:  # noqa: BLE001 - progress must never break exec
                _logger.debug("Progress callback raised", exc_info=True)
