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


def test_register_adds_agent() -> None:
    """Orchestrator should register agents by name."""
    orchestrator = Orchestrator()
    orchestrator.register(PlannerAgent())

    assert "planner" in orchestrator.agents


def test_duplicate_agent_registration_raises() -> None:
    """Orchestrator should reject duplicate agent names."""
    orchestrator = Orchestrator([PlannerAgent()])

    with pytest.raises(AgentAlreadyRegisteredError):
        orchestrator.register(PlannerAgent())


def test_route_sends_task_to_matching_agent() -> None:
    """Orchestrator should route supported tasks to the matching agent."""
    orchestrator = Orchestrator([PlannerAgent()])
    result = orchestrator.route(AgentTask(task_type="plan"))

    assert result.success is True
    assert result.agent_name == "planner"


def test_route_without_matching_agent_raises() -> None:
    """Orchestrator should fail clearly when no agent supports a task."""
    orchestrator = Orchestrator([PlannerAgent()])

    with pytest.raises(NoAgentForTaskError):
        orchestrator.route(AgentTask(task_type="unknown"))


def test_unregister_removes_agent() -> None:
    """Orchestrator should dynamically remove registered agents."""
    orchestrator = Orchestrator([PlannerAgent()])

    removed = orchestrator.unregister("planner")

    assert removed.name == "planner"
    assert "planner" not in orchestrator.agents


def test_unregister_missing_agent_raises() -> None:
    """Orchestrator should fail clearly when removing an unknown agent."""
    orchestrator = Orchestrator()

    with pytest.raises(AgentNotRegisteredError):
        orchestrator.unregister("missing")


def test_orchestrator_lifecycle_and_health_check() -> None:
    """Orchestrator should manage agent lifecycle methods."""
    async def run() -> dict[str, object]:
        orchestrator = Orchestrator([PlannerAgent()])
        await orchestrator.start()
        health = await orchestrator.health_check()
        await orchestrator.stop()
        return health

    health = asyncio.run(run())

    assert health["initialized"] is True
    assert health["started"] is True
    assert health["agent_count"] == 1
    assert health["task_queue_running"] is True


def test_orchestrator_enqueue_processes_task() -> None:
    """Orchestrator should process tasks through the async queue."""
    async def run() -> bool:
        orchestrator = Orchestrator([PlannerAgent()])
        await orchestrator.start()
        task_id = await orchestrator.enqueue(AgentTask(task_type="plan"))
        await orchestrator.task_queue.join()
        result = orchestrator.task_queue.get_result(task_id)
        await orchestrator.stop()
        return result is not None and result.success

    assert asyncio.run(run()) is True
