"""Base abstractions for Jarvis agents."""

from abc import abstractmethod
from collections.abc import Iterable
from typing import final

from agents.contracts import AgentMessage, AgentResult, AgentTask
from agents.interfaces import IAgent
from orchestrator.interfaces import IEventBus, ISharedContext


class Agent(IAgent):
    """Abstract base class for all Jarvis agents.

    Provides shared lifecycle management (initialize, start, stop,
    health_check), runtime binding, and message publishing.  Subclasses
    must implement :meth:`handle`.
    """

    def __init__(self, name: str, supported_task_types: Iterable[str]) -> None:
        self._name = name
        self._supported_task_types = frozenset(supported_task_types)
        self._context: ISharedContext | None = None
        self._event_bus: IEventBus | None = None
        self._initialized = False
        self._started = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_task_types(self) -> frozenset[str]:
        return self._supported_task_types

    @property
    def context(self) -> ISharedContext:
        if self._context is None:
            raise RuntimeError("Agent context has not been assigned.")
        return self._context

    @property
    def event_bus(self) -> IEventBus:
        if self._event_bus is None:
            raise RuntimeError("Agent event bus has not been assigned.")
        return self._event_bus

    def bind_runtime(self, context: ISharedContext, event_bus: IEventBus) -> None:
        self._context = context
        self._event_bus = event_bus

    @final
    def can_handle(self, task: AgentTask) -> bool:
        return task.task_type in self._supported_task_types

    async def initialize(self) -> None:
        self._initialized = True

    async def start(self) -> None:
        if not self._initialized:
            await self.initialize()
        self._started = True

    async def stop(self) -> None:
        self._started = False

    async def health_check(self) -> dict[str, object]:
        return {
            "name": self.name,
            "initialized": self._initialized,
            "started": self._started,
            "supported_task_types": sorted(self.supported_task_types),
        }

    async def publish(self, topic: str, payload: dict[str, object]) -> AgentMessage:
        message = AgentMessage(topic=topic, sender=self.name, payload=dict(payload))
        await self.event_bus.publish(message)
        return message

    @abstractmethod
    async def handle(self, task: AgentTask) -> AgentResult:
        """Handle a task and return a structured result."""
