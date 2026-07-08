"""Tests for planning.telemetry and its integration into the executor/coordinator.

Covers (cycle 8, telemetry):
* the sinks (null / in-memory / logging) and the non-raising facade;
* event JSON validity;
* that the TaskExecutor emits started/retry/completed with real latency/attempts;
* that the PlanningCoordinator emits plan_decided / plan_declined / run_completed
  on the right paths, including fallback reasons;
* that telemetry is a true no-op when disabled (default) and never raises;
* the build_planning_subsystem factory honouring PLANNING_TELEMETRY_ENABLED.

All offline — no Ollama, no network, no ChromaDB.
"""

from __future__ import annotations

import asyncio
import json

from config.settings import Settings
from planning import build_planning_subsystem
from planning.coordinator import PlanningCoordinator
from planning.executor import TaskExecutor
from planning.models import NodeResult, Plan, RetryPolicy, TaskNode, TaskStatus
from planning.telemetry import (
    InMemoryTelemetrySink,
    LoggingTelemetrySink,
    NullTelemetrySink,
    PlanningTelemetry,
    TelemetryEvent,
    TelemetryKind,
    build_telemetry_sink,
)
from planning.verifier import ResponseVerifier


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class ScriptedInvoker:
    """Invoker whose behaviour per node id is scripted (mirrors executor tests)."""

    def __init__(self, behaviours: dict | None = None) -> None:
        self._behaviours = behaviours or {}

    async def invoke(self, node: TaskNode, context: str = "") -> NodeResult:
        behaviour = self._behaviours.get(node.id, "ok")
        if behaviour == "ok":
            return NodeResult(node.id, TaskStatus.SUCCEEDED, output=f"{node.id}-out",
                              backend="reasoning", attempts=1)
        if behaviour == "fail":
            return NodeResult(node.id, TaskStatus.FAILED, error="nope",
                              backend="reasoning")
        if callable(behaviour):
            return await behaviour(node)
        return NodeResult(node.id, TaskStatus.SUCCEEDED)


class StubPlanner:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    async def decompose(self, goal, memory_context, catalog) -> Plan:
        return self._plan


class FakeKnowledge:
    available = True

    async def answer(self, text: str) -> str:
        return "A clear synthesized answer."


def _plan(nodes: list[TaskNode], conf: float = 0.8) -> Plan:
    return Plan(goal="g", nodes=tuple(nodes), overall_confidence=conf)


def _single_reasoning_plan(conf: float = 0.8) -> Plan:
    node = TaskNode(id="s1", description="explain the topic clearly",
                    required_tool="reasoning", confidence=conf)
    return Plan(goal="explain the topic", nodes=(node,), overall_confidence=conf)


# ----------------------------------------------------------------------
# Event + sinks
# ----------------------------------------------------------------------

class TestTelemetryEvent:
    def test_to_dict_flattens_fields(self) -> None:
        e = TelemetryEvent(kind="task_completed", timestamp=1.23456,
                           fields={"node_id": "a", "duration_ms": 12.5})
        d = e.to_dict()
        assert d["event"] == "task_completed"
        assert d["ts"] == 1.235  # rounded to 3dp
        assert d["node_id"] == "a"
        assert d["duration_ms"] == 12.5

    def test_to_json_is_valid_single_object(self) -> None:
        e = TelemetryEvent(kind="plan_decided", timestamp=0.0,
                           fields={"strategy": "llm", "node_count": 3})
        parsed = json.loads(e.to_json())
        assert parsed["event"] == "plan_decided"
        assert parsed["node_count"] == 3

    def test_to_json_survives_nonserialisable_field(self) -> None:
        e = TelemetryEvent(kind="x", timestamp=0.0, fields={"obj": object()})
        # Must not raise; the object is coerced via str.
        parsed = json.loads(e.to_json())
        assert "obj" in parsed


class TestSinks:
    def test_null_sink_discards(self) -> None:
        sink = NullTelemetrySink()
        sink.emit(TelemetryEvent("x", 0.0))  # no error, nothing retained

    def test_in_memory_sink_collects_and_filters(self) -> None:
        sink = InMemoryTelemetrySink()
        sink.emit(TelemetryEvent(TelemetryKind.TASK_STARTED.value, 0.0))
        sink.emit(TelemetryEvent(TelemetryKind.TASK_COMPLETED.value, 0.0))
        sink.emit(TelemetryEvent(TelemetryKind.TASK_COMPLETED.value, 0.0))
        assert len(sink.events) == 3
        assert len(sink.of_kind(TelemetryKind.TASK_COMPLETED)) == 2
        assert len(sink.of_kind("task_started")) == 1

    def test_in_memory_sink_bounded(self) -> None:
        sink = InMemoryTelemetrySink(max_events=5)
        for _ in range(20):
            sink.emit(TelemetryEvent("x", 0.0))
        assert len(sink.events) == 5

    def test_logging_sink_writes_json_line(self, caplog) -> None:
        sink = LoggingTelemetrySink()
        with caplog.at_level("INFO", logger="planning.telemetry"):
            sink.emit(TelemetryEvent("task_completed", 1.0, {"node_id": "a"}))
        assert any(json.loads(rec.message)["event"] == "task_completed"
                   for rec in caplog.records)

    def test_build_sink_factory(self) -> None:
        assert isinstance(build_telemetry_sink(True), LoggingTelemetrySink)
        assert isinstance(build_telemetry_sink(False), NullTelemetrySink)


class TestFacade:
    def test_default_is_disabled_noop(self) -> None:
        t = PlanningTelemetry()
        assert t.enabled is False
        # Calling every method is a harmless no-op.
        t.plan_decided(goal="g", strategy="llm", node_count=1, confidence=0.8)
        t.plan_declined(goal="g", reason="empty_plan", confidence=0.0)
        t.task_started(node_id="a", tool="reasoning", description="d")
        t.task_retry(node_id="a", tool="reasoning", attempt=1, reason="x")
        t.task_completed(node_id="a", tool="reasoning", status="succeeded",
                         attempts=1, duration_ms=1.0)
        t.run_completed(goal="g", accepted=True, reason="ok", wall_time_ms=1.0,
                        succeeded=1, total=1)

    def test_enabled_emits_to_sink(self) -> None:
        sink = InMemoryTelemetrySink()
        t = PlanningTelemetry(sink, clock=lambda: 42.0)
        t.plan_decided(goal="g", strategy="llm", node_count=2, confidence=0.9)
        assert t.enabled is True
        assert len(sink.events) == 1
        ev = sink.events[0]
        assert ev.kind == "plan_decided"
        assert ev.timestamp == 42.0
        assert ev.fields["node_count"] == 2

    def test_facade_never_raises_on_bad_sink(self) -> None:
        from planning.telemetry import ITelemetrySink

        class Boom(ITelemetrySink):
            def emit(self, event):
                raise RuntimeError("sink down")

        t = PlanningTelemetry(Boom())
        # enabled is True (a real, non-null sink)…
        assert t.enabled is True
        # …but a raising sink must never propagate out of the facade.
        t.task_started(node_id="a", tool="reasoning", description="d")
        t.run_completed(goal="g", accepted=True, reason="ok", wall_time_ms=1.0,
                        succeeded=1, total=1)

    def test_goal_is_clipped(self) -> None:
        sink = InMemoryTelemetrySink()
        t = PlanningTelemetry(sink)
        t.plan_decided(goal="x" * 500, strategy="llm", node_count=1, confidence=0.5)
        assert len(sink.events[0].fields["goal"]) <= 200


# ----------------------------------------------------------------------
# Executor integration
# ----------------------------------------------------------------------

class TestExecutorTelemetry:
    async def test_started_and_completed_emitted(self) -> None:
        sink = InMemoryTelemetrySink()
        t = PlanningTelemetry(sink)
        node = TaskNode(id="a", description="do a", required_tool="reasoning")
        ex = TaskExecutor(ScriptedInvoker(), task_timeout_seconds=5, telemetry=t)
        await ex.execute(_plan([node]))
        assert len(sink.of_kind(TelemetryKind.TASK_STARTED)) == 1
        completed = sink.of_kind(TelemetryKind.TASK_COMPLETED)
        assert len(completed) == 1
        assert completed[0].fields["status"] == "succeeded"
        assert completed[0].fields["tool"] == "reasoning"
        assert completed[0].fields["attempts"] == 1
        assert "duration_ms" in completed[0].fields

    async def test_retry_emitted_with_attempt_number(self) -> None:
        sink = InMemoryTelemetrySink()
        t = PlanningTelemetry(sink)
        node = TaskNode(id="a", description="a",
                        retry_policy=RetryPolicy(max_retries=2, backoff_seconds=0.0))
        ex = TaskExecutor(ScriptedInvoker({"a": "fail"}), task_timeout_seconds=5,
                          telemetry=t)
        await ex.execute(_plan([node]))
        retries = sink.of_kind(TelemetryKind.TASK_RETRY)
        assert len(retries) == 2  # 2 retries between 3 attempts
        assert retries[0].fields["attempt"] == 1
        assert retries[1].fields["attempt"] == 2
        # Terminal completion still recorded exactly once.
        assert len(sink.of_kind(TelemetryKind.TASK_COMPLETED)) == 1

    async def test_completed_records_failure_status_and_error(self) -> None:
        sink = InMemoryTelemetrySink()
        t = PlanningTelemetry(sink)
        node = TaskNode(id="a", description="a", retry_policy=RetryPolicy.none())
        ex = TaskExecutor(ScriptedInvoker({"a": "fail"}), task_timeout_seconds=5,
                          telemetry=t)
        await ex.execute(_plan([node]))
        completed = sink.of_kind(TelemetryKind.TASK_COMPLETED)[0]
        assert completed.fields["status"] == "failed"
        assert completed.fields["error"]

    async def test_skipped_nodes_do_not_emit_task_events(self) -> None:
        # a fails (critical) → b skipped. b never started/completed telemetry.
        sink = InMemoryTelemetrySink()
        t = PlanningTelemetry(sink)
        nodes = [
            TaskNode(id="a", description="a", retry_policy=RetryPolicy.none()),
            TaskNode(id="b", description="b", dependencies=["a"]),
        ]
        ex = TaskExecutor(ScriptedInvoker({"a": "fail"}), task_timeout_seconds=5,
                          telemetry=t)
        await ex.execute(_plan(nodes))
        started_ids = {e.fields["node_id"] for e in sink.of_kind(TelemetryKind.TASK_STARTED)}
        assert started_ids == {"a"}  # b was skipped, never started

    async def test_executor_default_telemetry_is_noop(self) -> None:
        # No telemetry arg → constructed with a null facade → no error, behaves same.
        node = TaskNode(id="a", description="a", required_tool="reasoning")
        ex = TaskExecutor(ScriptedInvoker(), task_timeout_seconds=5)
        metrics = await ex.execute(_plan([node]))
        assert metrics.succeeded == 1


# ----------------------------------------------------------------------
# Coordinator integration
# ----------------------------------------------------------------------

class TestCoordinatorTelemetry:
    async def test_accepted_run_emits_decided_and_completed(self) -> None:
        sink = InMemoryTelemetrySink()
        coord = PlanningCoordinator(
            StubPlanner(_single_reasoning_plan()),
            ResponseVerifier(),
            knowledge_engine=FakeKnowledge(),
            telemetry=PlanningTelemetry(sink),
        )
        out = await coord.run("explain the topic")
        assert out.accepted
        assert len(sink.of_kind(TelemetryKind.PLAN_DECIDED)) == 1
        run_events = sink.of_kind(TelemetryKind.RUN_COMPLETED)
        assert len(run_events) == 1
        rc = run_events[0].fields
        assert rc["accepted"] is True
        assert rc["reason"] == "ok"
        assert rc["succeeded"] == 1
        assert rc["total"] == 1
        assert "wall_time_ms" in rc

    async def test_low_confidence_emits_declined_and_completed(self) -> None:
        sink = InMemoryTelemetrySink()
        coord = PlanningCoordinator(
            StubPlanner(_single_reasoning_plan(conf=0.1)),
            ResponseVerifier(),
            knowledge_engine=FakeKnowledge(),
            min_goal_confidence=0.5,
            telemetry=PlanningTelemetry(sink),
        )
        out = await coord.run("do something")
        assert not out.accepted
        declined = sink.of_kind(TelemetryKind.PLAN_DECLINED)
        assert len(declined) == 1
        assert declined[0].fields["reason"] == "low_plan_confidence"
        # A run_completed with the fallback reason is still emitted.
        rc = sink.of_kind(TelemetryKind.RUN_COMPLETED)[0].fields
        assert rc["accepted"] is False
        assert rc["reason"] == "low_plan_confidence"

    async def test_empty_plan_emits_declined(self) -> None:
        sink = InMemoryTelemetrySink()
        coord = PlanningCoordinator(
            StubPlanner(Plan(goal="x", nodes=(), overall_confidence=0.0)),
            ResponseVerifier(),
            telemetry=PlanningTelemetry(sink),
        )
        await coord.run("do something")
        assert len(sink.of_kind(TelemetryKind.PLAN_DECLINED)) == 1
        assert sink.of_kind(TelemetryKind.PLAN_DECLINED)[0].fields["reason"] == "empty_plan"

    async def test_coordinator_error_still_emits_run_completed(self) -> None:
        class Boom:
            async def decompose(self, *a, **k):
                raise RuntimeError("kaboom")

        sink = InMemoryTelemetrySink()
        coord = PlanningCoordinator(Boom(), ResponseVerifier(),
                                    telemetry=PlanningTelemetry(sink))
        out = await coord.run("x")
        assert not out.accepted and out.reason == "coordinator_error"
        rc = sink.of_kind(TelemetryKind.RUN_COMPLETED)
        assert len(rc) == 1
        assert rc[0].fields["reason"] == "coordinator_error"

    async def test_exactly_one_run_completed_per_run(self) -> None:
        sink = InMemoryTelemetrySink()
        coord = PlanningCoordinator(
            StubPlanner(_single_reasoning_plan()),
            ResponseVerifier(),
            knowledge_engine=FakeKnowledge(),
            telemetry=PlanningTelemetry(sink),
        )
        await coord.run("explain the topic")
        await coord.run("explain the topic")
        assert len(sink.of_kind(TelemetryKind.RUN_COMPLETED)) == 2

    async def test_default_coordinator_telemetry_noop(self) -> None:
        # No telemetry arg → null facade → identical behaviour, no error.
        coord = PlanningCoordinator(
            StubPlanner(_single_reasoning_plan()),
            ResponseVerifier(),
            knowledge_engine=FakeKnowledge(),
        )
        out = await coord.run("explain the topic")
        assert out.accepted


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

class TestFactoryTelemetry:
    def test_telemetry_on_by_default(self) -> None:
        coord = build_planning_subsystem(Settings(planning_enabled=True))
        assert coord is not None and coord._telemetry.enabled is True

    def test_telemetry_can_be_disabled(self) -> None:
        coord = build_planning_subsystem(
            Settings(planning_enabled=True, planning_telemetry_enabled=False)
        )
        assert coord is not None and coord._telemetry.enabled is False
