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
    ) -> LLMResponse:
        from llm.interfaces import LLMResponse
        return LLMResponse(content="1. inspect\n2. execute")

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
    assert result.data["response"] == "1. inspect\n2. execute"


@pytest.mark.asyncio
async def test_memory_agent_handles_memory_tasks(tmp_path: str) -> None:
    from memory.document_store import SQLiteDocumentStore
    from memory.memory_manager import MemoryManager
    from memory.memory_service import MemoryService
    from memory.vector_store import ChromaVectorStore

    vs = ChromaVectorStore(str(tmp_path / "ma_v"))
    ds = SQLiteDocumentStore(str(tmp_path / "ma_d.db"))
    mm = MemoryManager(vector_store=vs, document_store=ds, importance_threshold=0.0)
    await mm.initialize()
    svc = MemoryService(mm)

    agent = MemoryAgent(memory_service=svc)
    result = await agent.handle(AgentTask(task_type="memory.retrieve", payload={"memory_id": "nonexistent"}))

    assert result.success is True
    assert result.agent_name == "memory"
    assert result.data["status"] == "not_found"


@pytest.mark.asyncio
async def test_tool_agent_handles_tool_tasks() -> None:
    result = await ToolAgent().handle(AgentTask(
        task_type="tool.execute",
        payload={"tool_name": "system_info", "arguments": {}},
    ))

    assert result.success is True
    assert result.agent_name == "tool"


@pytest.mark.asyncio
async def test_voice_agent_reports_unavailable_without_providers() -> None:
    # With no TTS/STT/audio wired in, the agent must degrade gracefully.
    result = await VoiceAgent().handle(
        AgentTask(task_type="voice.output", payload={"text": "hello"})
    )

    assert result.success is False
    assert result.agent_name == "voice"
    assert result.data["status"] == "unavailable"
