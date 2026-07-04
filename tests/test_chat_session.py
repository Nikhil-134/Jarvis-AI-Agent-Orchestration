"""Tests for chat session history management."""

from collections.abc import Iterable

from llm import BaseLLMProvider, ChatSession, LLMConfig


class EchoProvider(BaseLLMProvider):
    """Simple provider used by chat session tests."""

    @property
    def name(self) -> str:
        """Return provider name."""
        return "echo"

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return a deterministic response."""
        return f"response:{prompt}"

    def stream(self, prompt: str, system_prompt: str | None = None) -> Iterable[str]:
        """Return deterministic stream chunks."""
        return ("a", "b")


def test_chat_session_stores_user_and_assistant_history() -> None:
    """ChatSession should retain conversation history."""
    session = ChatSession(EchoProvider(LLMConfig(provider="echo", model="echo")))

    response = session.send("hello")

    assert response == "response:user: hello"
    assert [message.role for message in session.history] == ["user", "assistant"]


def test_chat_session_stream_stores_joined_assistant_history() -> None:
    """ChatSession should store streamed assistant chunks as one message."""
    session = ChatSession(EchoProvider(LLMConfig(provider="echo", model="echo")))

    chunks = session.stream("hello")

    assert chunks == ["a", "b"]
    assert session.history[-1].content == "ab"
