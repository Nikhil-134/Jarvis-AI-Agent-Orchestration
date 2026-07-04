"""Base abstractions for Jarvis agents."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import final

from agents.contracts import AgentMessage, AgentResult, AgentTask
from orchestrator.context import SharedContext
from orchestrator.message_bus import MessageBus


class Agent(ABC):
    """Abstract base class for all Jarvis agents."""

    def __init__(self, name: str, supported_task_types: Iterable[str]) -> None:
        """Initialize an agent with a unique name and supported task types."""
        self._name = name
        self._supported_task_types = frozenset(supported_task_types)
        self._context: SharedContext | None = None
        self._message_bus: MessageBus | None = None
        self._initialized = False
        self._started = False

    @property
    def name(self) -> str:
        """Return the agent name used by the orchestrator registry."""
        return self._name

    @property
    def supported_task_types(self) -> frozenset[str]:
        """Return task types this agent can handle."""
        return self._supported_task_types

    @property
    def context(self) -> SharedContext:
        """Return the shared context assigned by the orchestrator."""
        if self._context is None:
            raise RuntimeError("Agent context has not been assigned.")
        return self._context

    @property
    def message_bus(self) -> MessageBus:
        """Return the message bus assigned by the orchestrator."""
        if self._message_bus is None:
            raise RuntimeError("Agent message bus has not been assigned.")
        return self._message_bus

    def bind_runtime(self, context: SharedContext, message_bus: MessageBus) -> None:
        """Attach shared runtime dependencies to this agent."""
        self._context = context
        self._message_bus = message_bus

    @final
    def can_handle(self, task: AgentTask) -> bool:
        """Return whether this agent supports the supplied task."""
        return task.task_type in self._supported_task_types

    async def initialize(self) -> None:
        """Prepare the agent before it starts handling work."""
        self._initialized = True

    async def start(self) -> None:
        """Mark the agent as ready to handle work."""
        if not self._initialized:
            await self.initialize()
        self._started = True

    async def stop(self) -> None:
        """Stop the agent from handling work."""
        self._started = False

    async def health_check(self) -> dict[str, object]:
        """Return a basic health report for this agent."""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "started": self._started,
            "supported_task_types": sorted(self.supported_task_types),
        }

    async def publish(self, topic: str, payload: dict[str, object]) -> AgentMessage:
        """Publish an inter-agent message through the message bus."""
        message = AgentMessage(topic=topic, sender=self.name, payload=dict(payload))
        await self.message_bus.publish(message)
        return message

    @abstractmethod
    def handle(self, task: AgentTask) -> AgentResult:
        """Handle a task and return a structured result."""
