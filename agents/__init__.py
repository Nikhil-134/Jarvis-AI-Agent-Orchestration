"""Agent package exports."""

from typing import Any

__all__ = [
    "Agent",
    "AgentMessage",
    "AgentResult",
    "AgentTask",
    "MemoryAgent",
    "PlannerAgent",
    "ToolAgent",
    "VoiceAgent",
]


def __getattr__(name: str) -> Any:
    """Load agent exports lazily so modules remain runnable with python -m."""
    if name == "Agent":
        from agents.base import Agent

        return Agent
    if name in {"AgentMessage", "AgentResult", "AgentTask"}:
        from agents.contracts import AgentMessage, AgentResult, AgentTask

        return {
            "AgentMessage": AgentMessage,
            "AgentResult": AgentResult,
            "AgentTask": AgentTask,
        }[name]
    if name == "MemoryAgent":
        from agents.memory_agent import MemoryAgent

        return MemoryAgent
    if name == "PlannerAgent":
        from agents.planner import PlannerAgent

        return PlannerAgent
    if name == "ToolAgent":
        from agents.tool_agent import ToolAgent

        return ToolAgent
    if name == "VoiceAgent":
        from agents.voice_agent import VoiceAgent

        return VoiceAgent
    raise AttributeError(f"module 'agents' has no attribute {name!r}")
