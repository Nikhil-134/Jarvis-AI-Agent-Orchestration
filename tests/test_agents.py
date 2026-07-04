"""Tests for Phase 1 agent implementations."""

from collections.abc import Iterable

from agents import AgentTask, MemoryAgent, PlannerAgent, ToolAgent, VoiceAgent
from llm import BaseLLMProvider, LLMConfig


def test_planner_agent_handles_plan_tasks() -> None:
    """PlannerAgent should accept plan tasks."""
    result = PlannerAgent().handle(AgentTask(task_type="plan"))

    assert result.success is True
    assert result.agent_name == "planner"


class StaticPlannerProvider(BaseLLMProvider):
    """Static provider used to verify planner LLM integration."""

    @property
    def name(self) -> str:
        """Return provider name."""
        return "static"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return a fixed plan."""
        return "1. inspect\n2. execute"

    def stream(self, prompt: str, system_prompt: str | None = None) -> Iterable[str]:
        """Return fixed chunks."""
        return ("1. inspect", "\n2. execute")


def test_planner_agent_uses_llm_provider_when_configured() -> None:
    """PlannerAgent should call its LLM provider when supplied."""
    provider = StaticPlannerProvider(LLMConfig(provider="static", model="static"))
    result = PlannerAgent(llm_provider=provider).handle(
        AgentTask(task_type="plan", payload={"goal": "test"})
    )

    assert result.success is True
    assert result.data["status"] == "completed"
    assert result.data["plan"] == "1. inspect\n2. execute"


def test_memory_agent_handles_memory_tasks() -> None:
    """MemoryAgent should accept memory tasks."""
    result = MemoryAgent().handle(AgentTask(task_type="memory.retrieve"))

    assert result.success is True
    assert result.agent_name == "memory"


def test_tool_agent_handles_tool_tasks() -> None:
    """ToolAgent should accept tool tasks."""
    result = ToolAgent().handle(AgentTask(task_type="tool.execute"))

    assert result.success is True
    assert result.agent_name == "tool"


def test_voice_agent_handles_voice_tasks() -> None:
    """VoiceAgent should accept voice tasks."""
    result = VoiceAgent().handle(AgentTask(task_type="voice.output"))

    assert result.success is True
    assert result.agent_name == "voice"
