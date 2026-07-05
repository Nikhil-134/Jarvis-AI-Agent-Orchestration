"""Tests for agent implementations."""

from collections.abc import AsyncIterable

import pytest

from agents import AgentTask, MemoryAgent, PlannerAgent, ToolAgent, VoiceAgent
from llm import BaseLLMProvider, LLMConfig


@pytest.mark.asyncio
async def test_planner_agent_handles_plan_tasks() -> None:
    result = await PlannerAgent().handle(AgentTask(task_type="plan"))

    assert result.success is True
    assert result.agent_name == "planner"


class StaticPlannerProvider(BaseLLMProvider):
    """Static provider used to verify planner LLM integration."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(provider="static", model="static"))

    @property
    def name(self) -> str:
        return "static"

    async def _generate_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> str:
        return "1. inspect\n2. execute"

    async def _stream_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> AsyncIterable[str]:
        yield "1. inspect"
        yield "\n2. execute"


@pytest.mark.asyncio
async def test_planner_agent_uses_llm_provider_when_configured() -> None:
    provider = StaticPlannerProvider()
    result = await PlannerAgent(llm_provider=provider).handle(
        AgentTask(task_type="plan", payload={"goal": "test"})
    )

    assert result.success is True
    assert result.data["status"] == "completed"
    assert result.data["plan"] == "1. inspect\n2. execute"


@pytest.mark.asyncio
async def test_memory_agent_handles_memory_tasks() -> None:
    result = await MemoryAgent().handle(AgentTask(task_type="memory.retrieve"))

    assert result.success is True
    assert result.agent_name == "memory"


@pytest.mark.asyncio
async def test_tool_agent_handles_tool_tasks() -> None:
    result = await ToolAgent().handle(AgentTask(task_type="tool.execute"))

    assert result.success is True
    assert result.agent_name == "tool"


@pytest.mark.asyncio
async def test_voice_agent_handles_voice_tasks() -> None:
    result = await VoiceAgent().handle(AgentTask(task_type="voice.output"))

    assert result.success is True
    assert result.agent_name == "voice"
