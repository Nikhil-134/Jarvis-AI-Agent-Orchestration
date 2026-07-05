"""Tests for inter-agent message bus behavior."""

import asyncio

import pytest

from agents import AgentMessage
from orchestrator import MessageBus


@pytest.mark.asyncio
async def test_message_bus_publishes_to_async_subscribers() -> None:
    received: list[AgentMessage] = []
    bus = MessageBus()

    async def handler(message: AgentMessage) -> None:
        received.append(message)

    message = AgentMessage(topic="agent.event", sender="planner", payload={"x": 1})
    await bus.subscribe("agent.event", handler)
    await bus.publish(message)

    assert len(received) == 1
    assert received[0].payload == {"x": 1}


@pytest.mark.asyncio
async def test_message_bus_unsubscribe_stops_delivery() -> None:
    received: list[AgentMessage] = []
    bus = MessageBus()

    def handler(message: AgentMessage) -> None:
        received.append(message)

    await bus.subscribe("topic", handler)
    await bus.unsubscribe("topic", handler)
    await bus.publish(AgentMessage(topic="topic", sender="test"))

    assert received == []
