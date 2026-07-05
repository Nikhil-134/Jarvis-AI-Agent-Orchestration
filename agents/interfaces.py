"""Agent interface definitions for Jarvis."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from agents.contracts import AgentMessage, AgentResult, AgentTask
from orchestrator.interfaces import IEventBus, ISharedContext


class IAgent(ABC):
    """Interface all Jarvis agents must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent name used by the orchestrator registry."""

    @property
    @abstractmethod
    def supported_task_types(self) -> frozenset[str]:
        """Return task types this agent can handle."""

    @abstractmethod
    def bind_runtime(self, context: ISharedContext, event_bus: IEventBus) -> None:
        """Attach shared runtime dependencies to this agent."""

    @abstractmethod
    def can_handle(self, task: AgentTask) -> bool:
        """Return whether this agent supports the supplied task."""

    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the agent before it starts handling work."""

    @abstractmethod
    async def start(self) -> None:
        """Mark the agent as ready to handle work."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the agent from handling work."""

    @abstractmethod
    async def health_check(self) -> dict[str, object]:
        """Return a health report for this agent."""

    @abstractmethod
    async def publish(self, topic: str, payload: dict[str, object]) -> AgentMessage:
        """Publish an inter-agent message through the event bus."""

    @abstractmethod
    async def handle(self, task: AgentTask) -> AgentResult:
        """Handle a task and return a structured result."""
