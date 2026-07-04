"""Tests for inter-agent message bus behavior."""

import asyncio

from agents import AgentMessage
from orchestrator import MessageBus


def test_message_bus_publishes_to_async_subscribers() -> None:
    """MessageBus should deliver messages to subscribed async handlers."""
    received: list[AgentMessage] = []

    async def run() -> None:
        bus = MessageBus()

        async def handler(message: AgentMessage) -> None:
            received.append(message)

        message = AgentMessage(topic="agent.event", sender="planner", payload={"x": 1})
        await bus.subscribe("agent.event", handler)
        await bus.publish(message)

    asyncio.run(run())

    assert len(received) == 1
    assert received[0].payload == {"x": 1}


def test_message_bus_unsubscribe_stops_delivery() -> None:
    """MessageBus should stop delivering after unsubscribe."""
    received: list[AgentMessage] = []

    async def run() -> None:
        bus = MessageBus()

        def handler(message: AgentMessage) -> None:
            received.append(message)

        await bus.subscribe("topic", handler)
        await bus.unsubscribe("topic", handler)
        await bus.publish(AgentMessage(topic="topic", sender="test"))

    asyncio.run(run())

    assert received == []
