"""Chat session management with conversation history."""

import logging
from dataclasses import dataclass

from llm.base import BaseLLMProvider

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message in a chat conversation."""

    role: str
    content: str


class ChatSession:
    """Maintains conversation history around an LLM provider.

    Usage::

        session = ChatSession(provider, system_prompt="You are a helpful assistant.")
        reply = await session.send("Hello!")
        async for chunk in session.stream("Tell me more."):
            print(chunk, end="")
    """

    def __init__(self, provider: BaseLLMProvider, system_prompt: str | None = None) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """Return immutable conversation history."""
        return tuple(self._history)

    async def send(self, prompt: str) -> str:
        """Send a prompt, store user and assistant messages, return the response."""
        self._history.append(ChatMessage(role="user", content=prompt))
        response = await self.provider.generate(self._conversation_prompt(), self.system_prompt)
        self._history.append(ChatMessage(role="assistant", content=response))
        _logger.debug("Chat session send: %d messages in history", len(self._history))
        return response

    async def stream(self, prompt: str) -> list[str]:
        """Stream a prompt response and store the final assistant message."""
        self._history.append(ChatMessage(role="user", content=prompt))
        chunks: list[str] = []
        async for chunk in self.provider.stream(self._conversation_prompt(), self.system_prompt):
            chunks.append(chunk)
        self._history.append(ChatMessage(role="assistant", content="".join(chunks)))
        _logger.debug("Chat session stream: %d chunks, %d messages", len(chunks), len(self._history))
        return chunks

    def clear(self) -> None:
        """Clear all conversation history."""
        self._history.clear()
        _logger.debug("Chat session history cleared")

    def _conversation_prompt(self) -> str:
        """Render history into a provider-neutral prompt string."""
        return "\n".join(f"{message.role}: {message.content}" for message in self._history)
