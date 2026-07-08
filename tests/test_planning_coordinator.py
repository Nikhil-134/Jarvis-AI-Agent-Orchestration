"""Tests for planning.coordinator.PlanningCoordinator + build factory."""

from __future__ import annotations

from config.settings import Settings
from planning import build_planning_subsystem
from planning.coordinator import PlanningCoordinator
from planning.models import Plan, TaskNode
from planning.verifier import ResponseVerifier


class StubPlanner:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    async def decompose(self, goal, memory_context, catalog) -> Plan:
        return self._plan


class FakeKnowledge:
    def __init__(self, available: bool = True) -> None:
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    async def answer(self, text: str) -> str:
        return "A clear, synthesized answer to the request."


def _single_reasoning_plan(conf: float = 0.8) -> Plan:
    node = TaskNode(id="s1", description="explain the topic clearly",
                    required_tool="reasoning", confidence=conf)
    return Plan(goal="explain the topic", nodes=(node,), overall_confidence=conf)


class TestCoordinatorRun:
    async def test_happy_path_accepted(self) -> None:
        coord = PlanningCoordinator(
            StubPlanner(_single_reasoning_plan()),
            ResponseVerifier(),
            knowledge_engine=FakeKnowledge(available=True),
        )
        out = await coord.run("explain the topic")
        assert out.accepted
        assert "synthesized answer" in out.response
        assert out.metrics is not None and out.metrics.succeeded == 1

    async def test_empty_goal_declined(self) -> None:
        coord = PlanningCoordinator(StubPlanner(_single_reasoning_plan()),
                                    ResponseVerifier())
        out = await coord.run("   ")
        assert not out.accepted
        assert out.reason == "empty_goal"

    async def test_empty_plan_declined(self) -> None:
        empty = Plan(goal="x", nodes=(), overall_confidence=0.0)
        coord = PlanningCoordinator(StubPlanner(empty), ResponseVerifier())
        out = await coord.run("do something")
        assert not out.accepted
        assert out.reason == "empty_plan"

    async def test_low_confidence_plan_declined(self) -> None:
        coord = PlanningCoordinator(
            StubPlanner(_single_reasoning_plan(conf=0.1)),
            ResponseVerifier(),
            knowledge_engine=FakeKnowledge(),
            min_goal_confidence=0.5,
        )
        out = await coord.run("do something")
        assert not out.accepted
        assert out.reason == "low_plan_confidence"

    async def test_all_failed_declines_for_fallback(self) -> None:
        # No reasoning backend → the reasoning task fails → decline (fallback).
        coord = PlanningCoordinator(
            StubPlanner(_single_reasoning_plan(conf=0.8)),
            ResponseVerifier(),
            knowledge_engine=None,
            llm_guard=None,
        )
        out = await coord.run("do something")
        assert not out.accepted
        assert out.reason == "all_failed"

    async def test_never_raises(self) -> None:
        class Boom:
            async def decompose(self, *a, **k):
                raise RuntimeError("kaboom")

        coord = PlanningCoordinator(Boom(), ResponseVerifier())
        out = await coord.run("x")
        assert not out.accepted
        assert out.reason == "coordinator_error"


class TestBuildFactory:
    def test_disabled_returns_none(self) -> None:
        s = Settings(planning_enabled=False)
        assert build_planning_subsystem(s) is None

    def test_enabled_returns_coordinator(self) -> None:
        s = Settings(planning_enabled=True)
        coord = build_planning_subsystem(s)
        assert isinstance(coord, PlanningCoordinator)

    def test_none_settings_defaults_on(self) -> None:
        coord = build_planning_subsystem(None)
        assert isinstance(coord, PlanningCoordinator)

    def test_auto_approve_enables_dangerous(self) -> None:
        s = Settings(planning_enabled=True, tool_auto_approve=True)
        coord = build_planning_subsystem(s)
        assert coord._allow_dangerous is True
