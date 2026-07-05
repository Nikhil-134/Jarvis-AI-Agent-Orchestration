"""Tests for the Jarvis orchestrator."""

import asyncio

import pytest

from agents import AgentTask, PlannerAgent
from orchestrator import (
    AgentAlreadyRegisteredError,
    AgentNotRegisteredError,
    NoAgentForTaskError,
    Orchestrator,
)


@pytest.mark.asyncio
async def test_register_adds_agent() -> None:
    orchestrator = Orchestrator()
    orchestrator.register(PlannerAgent())

    assert "planner" in orchestrator.agents


@pytest.mark.asyncio
async def test_duplicate_agent_registration_raises() -> None:
    orchestrator = Orchestrator([PlannerAgent()])

    with pytest.raises(AgentAlreadyRegisteredError):
        orchestrator.register(PlannerAgent())


@pytest.mark.asyncio
async def test_route_sends_task_to_matching_agent() -> None:
    orchestrator = Orchestrator([PlannerAgent()])
    result = await orchestrator.route(AgentTask(task_type="plan"))

    assert result.success is True
    assert result.agent_name == "planner"


@pytest.mark.asyncio
async def test_route_without_matching_agent_raises() -> None:
    orchestrator = Orchestrator([PlannerAgent()])

    with pytest.raises(NoAgentForTaskError):
        await orchestrator.route(AgentTask(task_type="unknown"))


@pytest.mark.asyncio
async def test_unregister_removes_agent() -> None:
    orchestrator = Orchestrator([PlannerAgent()])

    removed = orchestrator.unregister("planner")

    assert removed.name == "planner"
    assert "planner" not in orchestrator.agents


@pytest.mark.asyncio
async def test_unregister_missing_agent_raises() -> None:
    orchestrator = Orchestrator()

    with pytest.raises(AgentNotRegisteredError):
        orchestrator.unregister("missing")


@pytest.mark.asyncio
async def test_orchestrator_lifecycle_and_health_check() -> None:
    orchestrator = Orchestrator([PlannerAgent()])
    await orchestrator.start()
    health = await orchestrator.health_check()
    await orchestrator.stop()

    assert health["initialized"] is True
    assert health["started"] is True
    assert health["agent_count"] == 1
    assert health["task_queue_running"] is True


@pytest.mark.asyncio
async def test_orchestrator_enqueue_processes_task() -> None:
    orchestrator = Orchestrator([PlannerAgent()])
    await orchestrator.start()
    task_id = await orchestrator.enqueue(AgentTask(task_type="plan"))
    await orchestrator.task_queue.join()
    result = orchestrator.task_queue.get_result(task_id)
    await orchestrator.stop()

    assert result is not None
    assert result.success is True
