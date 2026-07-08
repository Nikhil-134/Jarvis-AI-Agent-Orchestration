from __future__ import annotations

from typing import Any

import pytest

from agents.contracts import AgentResult, AgentTask
from orchestrator.workflow import WorkflowEngine, WorkflowPlan, WorkflowStep


# ===================================================================
# WorkflowStep
# ===================================================================

class TestWorkflowStep:
    def test_creation_minimal(self) -> None:
        step = WorkflowStep(task_type="research")
        assert step.task_type == "research"
        assert step.payload == {}
        assert step.agent_name is None
        assert step.depends_on == []
        assert step.result is None
        assert step.step_id == "research"

    def test_creation_full(self) -> None:
        step = WorkflowStep(
            task_type="code.generate",
            payload={"language": "python"},
            agent_name="veronica",
            depends_on=["research"],
        )
        assert step.task_type == "code.generate"
        assert step.payload == {"language": "python"}
        assert step.agent_name == "veronica"
        assert step.depends_on == ["research"]

    def test_depends_on_defaults_to_empty(self) -> None:
        step = WorkflowStep(task_type="test.run", depends_on=None)
        assert step.depends_on == []


# ===================================================================
# WorkflowPlan
# ===================================================================

class TestWorkflowPlan:
    def test_creation(self) -> None:
        plan = WorkflowPlan(goal="Build feature X")
        assert plan.goal == "Build feature X"
        assert plan.steps == []

    def test_add_step(self) -> None:
        plan = WorkflowPlan(goal="test")
        step = WorkflowStep(task_type="research")
        returned = plan.add_step(step)
        assert returned is step
        assert len(plan.steps) == 1
        assert plan.steps[0] is step

    def test_add_multiple_steps(self) -> None:
        plan = WorkflowPlan(goal="test")
        plan.add_step(WorkflowStep(task_type="research"))
        plan.add_step(WorkflowStep(task_type="code.generate"))
        assert len(plan.steps) == 2

    def test_is_empty_true(self) -> None:
        plan = WorkflowPlan(goal="test")
        assert plan.is_empty is True

    def test_is_empty_false(self) -> None:
        plan = WorkflowPlan(goal="test")
        plan.add_step(WorkflowStep(task_type="research"))
        assert plan.is_empty is False

    def test_completed_steps(self) -> None:
        plan = WorkflowPlan(goal="test")
        s1 = WorkflowStep(task_type="research")
        s2 = WorkflowStep(task_type="code.generate")
        plan.add_step(s1)
        plan.add_step(s2)

        s1.result = AgentResult(
            agent_name="friday", task_id="1", success=True, message="ok",
        )
        assert plan.completed_steps == [s1]
        assert plan.failed_steps == []

    def test_failed_steps(self) -> None:
        plan = WorkflowPlan(goal="test")
        s1 = WorkflowStep(task_type="research")
        s2 = WorkflowStep(task_type="code.generate")
        plan.add_step(s1)
        plan.add_step(s2)

        s1.result = AgentResult(
            agent_name="friday", task_id="1", success=False, message="fail",
        )
        s2.result = AgentResult(
            agent_name="veronica", task_id="2", success=True, message="ok",
        )
        assert plan.failed_steps == [s1]
        assert plan.completed_steps == [s1, s2]

    def test_all_succeeded_true(self) -> None:
        plan = WorkflowPlan(goal="test")
        s1 = WorkflowStep(task_type="research")
        s2 = WorkflowStep(task_type="code.generate")
        plan.add_step(s1)
        plan.add_step(s2)

        s1.result = AgentResult(
            agent_name="friday", task_id="1", success=True, message="ok",
        )
        s2.result = AgentResult(
            agent_name="veronica", task_id="2", success=True, message="ok",
        )
        assert plan.all_succeeded is True

    def test_all_succeeded_false_when_some_fail(self) -> None:
        plan = WorkflowPlan(goal="test")
        s1 = WorkflowStep(task_type="research")
        s2 = WorkflowStep(task_type="code.generate")
        plan.add_step(s1)
        plan.add_step(s2)

        s1.result = AgentResult(
            agent_name="friday", task_id="1", success=True, message="ok",
        )
        s2.result = AgentResult(
            agent_name="veronica", task_id="2", success=False, message="fail",
        )
        assert plan.all_succeeded is False

    def test_all_succeeded_false_when_not_all_complete(self) -> None:
        plan = WorkflowPlan(goal="test")
        plan.add_step(WorkflowStep(task_type="research"))
        plan.add_step(WorkflowStep(task_type="code.generate"))
        assert plan.all_succeeded is False


# ===================================================================
# WorkflowEngine
# ===================================================================

async def _mock_route(task: AgentTask) -> AgentResult:
    return AgentResult(
        agent_name="mock_agent",
        task_id=task.task_id,
        success=True,
        message=f"Handled {task.task_type}",
    )


class TestWorkflowEngine:
    @pytest.fixture
    def engine(self) -> WorkflowEngine:
        return WorkflowEngine(route_fn=_mock_route)

    @pytest.mark.asyncio
    async def test_execute_empty_plan(self, engine: WorkflowEngine) -> None:
        plan = WorkflowPlan(goal="empty")
        results = await engine.execute(plan)
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_single_step(self, engine: WorkflowEngine) -> None:
        plan = WorkflowPlan(goal="single")
        plan.add_step(WorkflowStep(task_type="research", payload={"topic": "AI"}))
        results = await engine.execute(plan)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].agent_name == "mock_agent"
        assert results[0].message == "Handled research"

    @pytest.mark.asyncio
    async def test_execute_multiple_independent_steps(self, engine: WorkflowEngine) -> None:
        plan = WorkflowPlan(goal="parallel")
        plan.add_step(WorkflowStep(task_type="research"))
        plan.add_step(WorkflowStep(task_type="code.generate"))
        plan.add_step(WorkflowStep(task_type="test.run"))
        results = await engine.execute(plan)
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self, engine: WorkflowEngine) -> None:
        plan = WorkflowPlan(goal="sequential")
        s1 = plan.add_step(WorkflowStep(task_type="research", depends_on=[]))
        s2 = plan.add_step(WorkflowStep(task_type="code.generate", depends_on=["research"]))
        s3 = plan.add_step(WorkflowStep(task_type="test.run", depends_on=["code.generate"]))
        results = await engine.execute(plan)
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_stores_result_on_step(self, engine: WorkflowEngine) -> None:
        plan = WorkflowPlan(goal="store_result")
        step = plan.add_step(WorkflowStep(task_type="research"))
        await engine.execute(plan)
        assert step.result is not None
        assert step.result.success is True

    @pytest.mark.asyncio
    async def test_route_fn_receives_correct_task(self, engine: WorkflowEngine) -> None:
        received: list[AgentTask] = []

        async def tracking_route(task: AgentTask) -> AgentResult:
            received.append(task)
            return AgentResult(
                agent_name="tracker",
                task_id=task.task_id,
                success=True,
                message="ok",
            )

        eng = WorkflowEngine(route_fn=tracking_route)
        plan = WorkflowPlan(goal="track")
        plan.add_step(WorkflowStep(task_type="research", payload={"q": "test"}))
        await eng.execute(plan)
        assert len(received) == 1
        assert received[0].task_type == "research"
        assert received[0].payload == {"q": "test"}

    @pytest.mark.asyncio
    async def test_route_exception_returns_failed_result(self) -> None:
        async def failing_route(task: AgentTask) -> AgentResult:
            msg = f"Intentional failure for {task.task_type}"
            raise RuntimeError(msg)

        engine = WorkflowEngine(route_fn=failing_route)
        plan = WorkflowPlan(goal="fail")
        plan.add_step(WorkflowStep(task_type="research"))
        results = await engine.execute(plan)
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].agent_name == "workflow"

    @pytest.mark.asyncio
    async def test_execute_respects_unmet_dependencies(self, engine: WorkflowEngine) -> None:
        plan = WorkflowPlan(goal="deadlock")
        plan.add_step(WorkflowStep(task_type="research", depends_on=["nonexistent"]))
        results = await engine.execute(plan)
        assert results == []
