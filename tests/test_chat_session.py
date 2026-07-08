"""Tests for chat session history management."""

from collections.abc import AsyncIterable

import pytest

from llm import BaseLLMProvider, ChatSession, LLMConfig


class EchoProvider(BaseLLMProvider):
    """Simple provider used by chat session tests."""

    def __init__(self) -> None:
        super().__init__(LLMConfig(provider="echo", model="echo"))

    @property
    def name(self) -> str:
        return "echo"

    async def _generate_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> LLMResponse:
        from llm.interfaces import LLMResponse
        return LLMResponse(content=f"response:{prompt}")

    async def _stream_once(
        self, prompt: str, system_prompt: str | None, tools=None
    ) -> AsyncIterable[str]:
        yield "a"
        yield "b"


@pytest.mark.asyncio
async def test_chat_session_stores_user_and_assistant_history() -> None:
    session = ChatSession(EchoProvider())

    response = await session.send("hello")

    assert response.content == "response:user: hello"
    assert [message.role for message in session.history] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_chat_session_stream_stores_joined_assistant_history() -> None:
    session = ChatSession(EchoProvider())

    chunks = await session.stream("hello")

    assert chunks == ["a", "b"]
    assert session.history[-1].content == "ab"
