"""Publish/subscribe messaging for inter-agent communication."""

import asyncio
import inspect
from collections import defaultdict

from agents.contracts import AgentMessage
from orchestrator.interfaces import IEventBus, MessageHandler


class MessageBus(IEventBus):
    """In-process asynchronous publish/subscribe message bus.

    Implements :class:`IEventBus`.  Suitable for single-process
    deployments.  Swap for a Redis / NATS implementation behind the
    same interface when scaling horizontally.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        async with self._lock:
            self._subscribers[topic].append(handler)

    async def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        async with self._lock:
            handlers = self._subscribers.get(topic, [])
            if handler in handlers:
                handlers.remove(handler)
            if not handlers:
                self._subscribers.pop(topic, None)

    async def publish(self, message: AgentMessage) -> None:
        async with self._lock:
            handlers = tuple(self._subscribers.get(message.topic, ()))

        await asyncio.gather(*(self._dispatch(handler, message) for handler in handlers))

    async def _dispatch(self, handler: MessageHandler, message: AgentMessage) -> None:
        result = handler(message)
        if inspect.isawaitable(result):
            await result
