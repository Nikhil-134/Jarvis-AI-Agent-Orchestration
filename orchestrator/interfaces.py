"""Orchestration interface definitions for Jarvis."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from agents.contracts import AgentMessage, AgentResult, AgentTask

MessageHandler = Callable[[AgentMessage], None | Awaitable[None]]


class ISharedContext(ABC):
    """Interface for thread-safe shared context storage."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Return a value from the context."""

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set a value in the context."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a value from the context when it exists."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of all context data."""


class IEventBus(ABC):
    """Interface for inter-agent publish/subscribe messaging."""

    @abstractmethod
    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe a handler to a topic."""

    @abstractmethod
    async def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """Remove a handler from a topic subscription."""

    @abstractmethod
    async def publish(self, message: AgentMessage) -> None:
        """Publish a message to all subscribers for its topic."""


TaskHandler = Callable[[AgentTask], Awaitable[AgentResult]]


class ITaskQueue(ABC):
    """Interface for asynchronous task processing."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Return whether queue workers are active."""

    @abstractmethod
    async def start(self) -> None:
        """Start queue workers."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop queue workers."""

    @abstractmethod
    async def enqueue(self, task: AgentTask) -> str:
        """Add a task to the queue and return its task id."""

    @abstractmethod
    async def join(self) -> None:
        """Wait until all queued tasks have been processed."""

    @abstractmethod
    def get_result(self, task_id: str) -> AgentResult | None:
        """Return a processed task result by task id."""
