"""Orchestrator package exports."""

from typing import Any

__all__ = [
    "AgentAlreadyRegisteredError",
    "AgentNotRegisteredError",
    "IEventBus",
    "ISharedContext",
    "ITaskQueue",
    "MessageBus",
    "MiddlewarePipeline",
    "NoAgentForTaskError",
    "Orchestrator",
    "OrchestratorError",
    "SharedContext",
    "TaskQueue",
]


def __getattr__(name: str) -> Any:
    if name == "Orchestrator":
        from orchestrator.core import Orchestrator
        return Orchestrator
    if name in {"ISharedContext", "IEventBus", "ITaskQueue"}:
        from orchestrator.interfaces import IEventBus, ISharedContext, ITaskQueue
        return {
            "ISharedContext": ISharedContext,
            "IEventBus": IEventBus,
            "ITaskQueue": ITaskQueue,
        }[name]
    if name == "MiddlewarePipeline":
        from orchestrator.middleware import MiddlewarePipeline
        return MiddlewarePipeline
    if name == "SharedContext":
        from orchestrator.context import SharedContext
        return SharedContext
    if name == "MessageBus":
        from orchestrator.message_bus import MessageBus
        return MessageBus
    if name == "TaskQueue":
        from orchestrator.task_queue import TaskQueue
        return TaskQueue
    if name in {
        "AgentAlreadyRegisteredError",
        "AgentNotRegisteredError",
        "NoAgentForTaskError",
        "OrchestratorError",
    }:
        from orchestrator.exceptions import (
            AgentAlreadyRegisteredError,
            AgentNotRegisteredError,
            NoAgentForTaskError,
            OrchestratorError,
        )
        return {
            "AgentAlreadyRegisteredError": AgentAlreadyRegisteredError,
            "AgentNotRegisteredError": AgentNotRegisteredError,
            "NoAgentForTaskError": NoAgentForTaskError,
            "OrchestratorError": OrchestratorError,
        }[name]
    raise AttributeError(f"module 'orchestrator' has no attribute {name!r}")
