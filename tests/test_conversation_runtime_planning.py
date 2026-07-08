"""Tests for ConversationRuntime ↔ Planning coordinator integration (cycle 8)."""

from __future__ import annotations

from planning.models import ExecutionMetrics, PlanningOutcome, VerificationResult
from runtime.conversation_runtime import ConversationRuntime
from runtime.intent_engine import Intent, IntentResult


class FakeCoordinator:
    def __init__(self, outcome: PlanningOutcome) -> None:
        self._outcome = outcome
        self.calls: list[tuple[str, str]] = []

    async def run(self, goal: str, memory_context: str = "") -> PlanningOutcome:
        self.calls.append((goal, memory_context))
        return self._outcome


def _accepted(response: str = "planned answer") -> PlanningOutcome:
    return PlanningOutcome(
        accepted=True, response=response,
        metrics=ExecutionMetrics(total=2, succeeded=2),
        verification=VerificationResult(ok=True, response=response, confidence=0.8),
        reason="ok",
    )


def _declined() -> PlanningOutcome:
    return PlanningOutcome(accepted=False, response="", reason="low_plan_confidence")


def _runtime() -> ConversationRuntime:
    # No orchestrator/LLM → routing + knowledge are inert; we drive intents
    # directly and stub the pieces we assert on.
    return ConversationRuntime(orchestrator=None)


class TestShouldPlan:
    def test_compound_actionable_plans(self) -> None:
        rt = _runtime()
        rt.set_planning_coordinator(FakeCoordinator(_accepted()))
        intent = IntentResult(
            primary=Intent("browser", 0.9),
            secondary=[Intent("knowledge_question", 0.8)],
            goal="search and summarize", requires_browser=True,
        )
        assert rt._should_plan(intent) is True

    def test_heavy_planning_intent_plans(self) -> None:
        rt = _runtime()
        rt.set_planning_coordinator(FakeCoordinator(_accepted()))
        intent = IntentResult(primary=Intent("coding", 0.9), goal="x",
                              requires_planning=True)
        assert rt._should_plan(intent) is True

    def test_single_tool_does_not_plan(self) -> None:
        rt = _runtime()
        rt.set_planning_coordinator(FakeCoordinator(_accepted()))
        intent = IntentResult(primary=Intent("tool", 0.9), goal="calc",
                              requires_tool=True)
        assert rt._should_plan(intent) is False

    def test_knowledge_question_does_not_plan(self) -> None:
        rt = _runtime()
        rt.set_planning_coordinator(FakeCoordinator(_accepted()))
        intent = IntentResult(primary=Intent("knowledge_question", 0.9), goal="x")
        assert rt._should_plan(intent) is False

    def test_current_info_does_not_plan(self) -> None:
        rt = _runtime()
        rt.set_planning_coordinator(FakeCoordinator(_accepted()))
        intent = IntentResult(primary=Intent("current_info", 0.9),
                              secondary=[Intent("knowledge_question", 0.8)],
                              goal="weather", requires_browser=True)
        assert rt._should_plan(intent) is False

    def test_no_coordinator_never_plans(self) -> None:
        rt = _runtime()  # no coordinator injected
        intent = IntentResult(primary=Intent("coding", 0.9), goal="x",
                              requires_planning=True)
        assert rt._should_plan(intent) is False

    def test_conversation_does_not_plan(self) -> None:
        rt = _runtime()
        rt.set_planning_coordinator(FakeCoordinator(_accepted()))
        intent = IntentResult(primary=Intent("greeting", 0.9), goal="hi",
                              requires_conversation=True)
        assert rt._should_plan(intent) is False


class TestPipelineBranch:
    async def test_accepted_plan_short_circuits_routing(self, monkeypatch) -> None:
        rt = _runtime()
        coord = FakeCoordinator(_accepted("the planned answer"))
        rt.set_planning_coordinator(coord)

        # Force the intent classification to a compound actionable intent.
        intent = IntentResult(
            primary=Intent("coding", 0.9), secondary=[Intent("plan", 0.8)],
            goal="x", requires_planning=True,
        )
        monkeypatch.setattr(rt.intent_engine, "classify", lambda g: intent)

        # Routing must NOT be called when planning is accepted.
        called = {"routed": False}

        async def _fail_route(*a, **k):
            called["routed"] = True
            raise AssertionError("routing should not run")

        monkeypatch.setattr(rt.routing, "route", _fail_route)

        out = await rt.process("do a multi step thing")
        assert "planned answer" in out
        assert coord.calls  # coordinator was consulted
        assert called["routed"] is False

    async def test_declined_plan_falls_through_to_routing(self, monkeypatch) -> None:
        rt = _runtime()
        rt.set_planning_coordinator(FakeCoordinator(_declined()))

        intent = IntentResult(
            primary=Intent("coding", 0.9), secondary=[Intent("plan", 0.8)],
            goal="x", requires_planning=True,
        )
        monkeypatch.setattr(rt.intent_engine, "classify", lambda g: intent)

        routed = {"called": False}

        async def _route(intent_arg, goal_arg):
            routed["called"] = True
            from agents.contracts import AgentResult
            return AgentResult(agent_name="x", task_id="t", success=True,
                               message="", data={"response": "routed answer"})

        monkeypatch.setattr(rt.routing, "route", _route)

        out = await rt.process("do a multi step thing")
        assert routed["called"] is True
        assert "routed answer" in out
