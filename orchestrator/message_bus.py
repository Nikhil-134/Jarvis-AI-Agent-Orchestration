"""Publish/subscribe messaging for inter-agent communication."""

import asyncio
import inspect
from collections import defaultdict
from collections.abc import Awaitable, Callable

from agents.contracts import AgentMessage

MessageHandler = Callable[[AgentMessage], None | Awaitable[None]]


class MessageBus:
    """In-process asynchronous publish/subscribe message bus."""

    def __init__(self) -> None:
        """Initialize the message bus with no subscribers."""
        self._subscribers: dict[str, list[MessageHandler]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def subscribe(self, topic: str, handler: MessageHandler) -> None:
        """Subscribe a handler to a topic."""
        async with self._lock:
            self._subscribers[topic].append(handler)

    async def unsubscribe(self, topic: str, handler: MessageHandler) -> None:
        """Remove a handler from a topic subscription."""
        async with self._lock:
            handlers = self._subscribers.get(topic, [])
            if handler in handlers:
                handlers.remove(handler)
            if not handlers:
                self._subscribers.pop(topic, None)

    async def publish(self, message: AgentMessage) -> None:
        """Publish a message to all subscribers for its topic."""
        async with self._lock:
            handlers = tuple(self._subscribers.get(message.topic, ()))

        await asyncio.gather(*(self._dispatch(handler, message) for handler in handlers))

    async def _dispatch(self, handler: MessageHandler, message: AgentMessage) -> None:
        """Dispatch one message to one handler."""
        result = handler(message)
        if inspect.isawaitable(result):
            await result


if __name__ == "__main__":
    async def demo() -> None:
        bus = MessageBus()

        async def handler(message: AgentMessage) -> None:
            print(message)

        await bus.subscribe("demo", handler)
        await bus.publish(AgentMessage(topic="demo", sender="system", payload={"ok": True}))

    asyncio.run(demo())
